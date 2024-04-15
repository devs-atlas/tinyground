/*
 * eventually, tinyground will support a host of "preset" models
 *
 * these are things like xor, mnist
 *
 * some of these preset models will be able to perfectly make
 * predictions on the data like XOR, other's wont.
 *
 * this is less concrete than tensor testing - testing the ability of the
 * optimizer to actually train a neural net properly
 */

describe('Identity Test', () => {
  test('should always return true', () => {
    expect(true).toBe(true);
  });
});
