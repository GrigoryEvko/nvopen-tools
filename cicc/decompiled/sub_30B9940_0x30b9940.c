// Function: sub_30B9940
// Address: 0x30b9940
//
__int64 __fastcall sub_30B9940(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 result; // rax
  const void *v8; // r14

  v6 = *(unsigned int *)(a2 + 8);
  result = *(unsigned int *)(a1 + 8);
  v8 = *(const void **)a2;
  if ( result + v6 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), result + v6, 8u, a5, a6);
    result = *(unsigned int *)(a1 + 8);
  }
  if ( 8 * v6 )
  {
    memcpy((void *)(*(_QWORD *)a1 + 8 * result), v8, 8 * v6);
    result = *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 8) = result + v6;
  return result;
}
