// Function: sub_B1CB00
// Address: 0xb1cb00
//
__int64 __fastcall sub_B1CB00(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 result; // rax
  const void *v4; // r14

  v2 = *(unsigned int *)(a2 + 8);
  result = *(unsigned int *)(a1 + 8);
  v4 = *(const void **)a2;
  if ( result + v2 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, a1 + 16, result + v2, 8);
    result = *(unsigned int *)(a1 + 8);
  }
  if ( 8 * v2 )
  {
    memcpy((void *)(*(_QWORD *)a1 + 8 * result), v4, 8 * v2);
    result = *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 8) = result + v2;
  return result;
}
