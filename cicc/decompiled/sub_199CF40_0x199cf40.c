// Function: sub_199CF40
// Address: 0x199cf40
//
__int64 *__fastcall sub_199CF40(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        _QWORD *a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 *result; // rax
  char v13; // dl
  __int64 *v14; // rsi
  unsigned int v15; // edi
  __int64 *v16; // rcx
  __int64 *v17; // [rsp-10h] [rbp-80h]
  char v20[96]; // [rsp+10h] [rbp-60h] BYREF

  if ( a7 )
  {
    result = (__int64 *)sub_199CBE0(a7, a2);
    if ( (_DWORD)result )
    {
      *(_QWORD *)a1 = -1;
      *(_QWORD *)(a1 + 8) = -1;
      *(_QWORD *)(a1 + 16) = -1;
      *(_QWORD *)(a1 + 24) = -1;
      return result;
    }
  }
  result = *(__int64 **)(a3 + 8);
  if ( *(__int64 **)(a3 + 16) != result )
    goto LABEL_4;
  v14 = &result[*(unsigned int *)(a3 + 28)];
  v15 = *(_DWORD *)(a3 + 28);
  if ( result == v14 )
  {
LABEL_18:
    if ( v15 < *(_DWORD *)(a3 + 24) )
    {
      *(_DWORD *)(a3 + 28) = v15 + 1;
      *v14 = a2;
      ++*(_QWORD *)a3;
LABEL_14:
      sub_199CCC0(a1, a2, a3, a4, a5, a6, a8);
      result = v17;
      if ( a7 )
      {
        if ( *(_DWORD *)(a1 + 4) == -1 )
          return (__int64 *)sub_199AF80((__int64)v20, a7, a2);
      }
      return result;
    }
LABEL_4:
    result = sub_16CCBA0(a3, a2);
    if ( !v13 )
      return result;
    goto LABEL_14;
  }
  v16 = 0;
  while ( a2 != *result )
  {
    if ( *result == -2 )
      v16 = result;
    if ( v14 == ++result )
    {
      if ( !v16 )
        goto LABEL_18;
      *v16 = a2;
      --*(_DWORD *)(a3 + 32);
      ++*(_QWORD *)a3;
      goto LABEL_14;
    }
  }
  return result;
}
