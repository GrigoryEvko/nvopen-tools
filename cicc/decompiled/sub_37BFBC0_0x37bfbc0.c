// Function: sub_37BFBC0
// Address: 0x37bfbc0
//
int *__fastcall sub_37BFBC0(__int64 a1)
{
  __int64 v1; // rdx
  int *result; // rax
  int *i; // rcx
  int v4; // edx

  v1 = *(unsigned int *)(a1 + 24);
  result = *(int **)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = &result[12 * v1]; result != i; result += 12 )
  {
    if ( result )
    {
      v4 = *result;
      *((_QWORD *)result + 2) = 0;
      *result = v4 & 0xFFF00000 | 0x15;
    }
  }
  return result;
}
