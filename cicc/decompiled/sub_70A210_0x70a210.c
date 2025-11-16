// Function: sub_70A210
// Address: 0x70a210
//
int *__fastcall sub_70A210(int *a1, char a2)
{
  char v4; // r9
  int *result; // rax
  unsigned int v6; // edx
  int v7; // esi

  v4 = 32 - a2;
  result = a1;
  do
  {
    v6 = result[1];
    v7 = *result++;
    *(result - 1) = (v7 << a2) | (v6 >> v4);
  }
  while ( result != a1 + 3 );
  a1[3] <<= a2;
  return result;
}
