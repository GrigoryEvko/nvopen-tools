// Function: sub_25A5A10
// Address: 0x25a5a10
//
unsigned __int64 __fastcall sub_25A5A10(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 result; // rax
  char v7; // dl

  if ( !*(_BYTE *)(a1 + 68) )
    goto LABEL_8;
  result = *(_QWORD *)(a1 + 48);
  a4 = *(unsigned int *)(a1 + 60);
  a3 = (__int64 *)(result + 8 * a4);
  if ( (__int64 *)result == a3 )
  {
LABEL_7:
    if ( (unsigned int)a4 < *(_DWORD *)(a1 + 56) )
    {
      *(_DWORD *)(a1 + 60) = a4 + 1;
      *a3 = a2;
      ++*(_QWORD *)(a1 + 40);
LABEL_9:
      result = *(unsigned int *)(a1 + 736);
      if ( result + 1 > *(unsigned int *)(a1 + 740) )
      {
        sub_C8D5F0(a1 + 728, (const void *)(a1 + 744), result + 1, 8u, a5, a6);
        result = *(unsigned int *)(a1 + 736);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 728) + 8 * result) = a2;
      ++*(_DWORD *)(a1 + 736);
      return result;
    }
LABEL_8:
    result = (unsigned __int64)sub_C8CC70(a1 + 40, a2, (__int64)a3, a4, a5, a6);
    if ( !v7 )
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
