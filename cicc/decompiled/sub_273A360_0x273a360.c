// Function: sub_273A360
// Address: 0x273a360
//
unsigned __int64 __fastcall sub_273A360(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  unsigned __int64 result; // rax
  char v9; // dl
  __int64 v10; // r12

  v7 = a1[1];
  if ( !*(_BYTE *)(v7 + 28) )
    goto LABEL_8;
  result = *(_QWORD *)(v7 + 8);
  a4 = *(unsigned int *)(v7 + 20);
  a3 = (__int64 *)(result + 8 * a4);
  if ( (__int64 *)result == a3 )
  {
LABEL_7:
    if ( (unsigned int)a4 < *(_DWORD *)(v7 + 16) )
    {
      *(_DWORD *)(v7 + 20) = a4 + 1;
      *a3 = a2;
      ++*(_QWORD *)v7;
LABEL_9:
      v10 = *a1;
      result = *(unsigned int *)(v10 + 8);
      if ( result + 1 > *(unsigned int *)(v10 + 12) )
      {
        sub_C8D5F0(v10, (const void *)(v10 + 16), result + 1, 8u, a5, a6);
        result = *(unsigned int *)(v10 + 8);
      }
      *(_QWORD *)(*(_QWORD *)v10 + 8 * result) = a2;
      ++*(_DWORD *)(v10 + 8);
      return result;
    }
LABEL_8:
    result = (unsigned __int64)sub_C8CC70(v7, a2, (__int64)a3, a4, a5, a6);
    if ( !v9 )
      return result;
    goto LABEL_9;
  }
  while ( a2 != *(_QWORD *)result )
  {
    result += 8LL;
    if ( a3 == (__int64 *)result )
      goto LABEL_7;
  }
  return result;
}
