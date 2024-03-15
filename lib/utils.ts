export function argsort(x: number[]): number[] {
  return x
    .map((value, index) => ({ value, index })) // Create an array of objects with value and index
    .sort((a, b) => a.value - b.value) // Sort by value
    .map(({ index }) => index); // Extract the sorted indices
}
