export function argsort(x: number[]): number[] {
  return x
    .map((value, index) => ({ value, index })) // Create an array of objects with value and index
    .sort((a, b) => a.value - b.value) // Sort by value
    .map(({ index }) => index); // Extract the sorted indices
}

export function isNDArray(data: any): data is NDArray {
  if (Array.isArray(data)) {
    // Every element must either be a number or an NDArray itself
    return data.every(element => typeof element === 'number' || isNDArray(element));
  }
  return false;
}

export type NDArray = number[] | NDArray[];
