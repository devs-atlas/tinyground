module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  // Optionally, specify the directory where Jest should look for tests
  roots: ['<rootDir>/src'],
  // Match test files
  testMatch: [
    '**/?(*.)+(spec|test).ts?(x)',
  ],
};
