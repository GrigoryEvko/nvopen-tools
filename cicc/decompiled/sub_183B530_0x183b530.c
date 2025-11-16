// Function: sub_183B530
// Address: 0x183b530
//
unsigned __int64 __fastcall sub_183B530(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned __int64 result; // rax
  char v9; // dl
  __int64 *v10; // rsi
  unsigned int v11; // edi
  __int64 *v12; // rcx

  result = *(_QWORD *)(a1 + 48);
  if ( *(_QWORD *)(a1 + 56) != result )
    goto LABEL_2;
  v10 = (__int64 *)(result + 8LL * *(unsigned int *)(a1 + 68));
  v11 = *(_DWORD *)(a1 + 68);
  if ( (__int64 *)result == v10 )
  {
LABEL_16:
    if ( v11 < *(_DWORD *)(a1 + 64) )
    {
      *(_DWORD *)(a1 + 68) = v11 + 1;
      *v10 = a2;
      ++*(_QWORD *)(a1 + 40);
LABEL_6:
      result = *(unsigned int *)(a1 + 744);
      if ( (unsigned int)result >= *(_DWORD *)(a1 + 748) )
      {
        sub_16CD150(a1 + 736, (const void *)(a1 + 752), 0, 8, a5, a6);
        result = *(unsigned int *)(a1 + 744);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 736) + 8 * result) = a2;
      ++*(_DWORD *)(a1 + 744);
      return result;
    }
LABEL_2:
    result = (unsigned __int64)sub_16CCBA0(a1 + 40, a2);
    if ( !v9 )
      return result;
    goto LABEL_6;
  }
  v12 = 0;
  while ( *(_QWORD *)result != a2 )
  {
    if ( *(_QWORD *)result == -2 )
      v12 = (__int64 *)result;
    result += 8LL;
    if ( v10 == (__int64 *)result )
    {
      if ( !v12 )
        goto LABEL_16;
      *v12 = a2;
      --*(_DWORD *)(a1 + 72);
      ++*(_QWORD *)(a1 + 40);
      goto LABEL_6;
    }
  }
  return result;
}
