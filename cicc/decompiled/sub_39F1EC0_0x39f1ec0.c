// Function: sub_39F1EC0
// Address: 0x39f1ec0
//
__int64 *__fastcall sub_39F1EC0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 *result; // rax
  __int64 *v5; // rsi
  unsigned int v6; // r8d
  __int64 *v7; // rcx

  v3 = *(_QWORD *)(a1 + 264);
  result = *(__int64 **)(v3 + 192);
  if ( *(__int64 **)(v3 + 200) != result )
  {
LABEL_2:
    result = sub_16CCBA0(v3 + 184, a2);
    goto LABEL_3;
  }
  v5 = &result[*(unsigned int *)(v3 + 212)];
  v6 = *(_DWORD *)(v3 + 212);
  if ( result == v5 )
  {
LABEL_12:
    if ( v6 < *(_DWORD *)(v3 + 208) )
    {
      *(_DWORD *)(v3 + 212) = v6 + 1;
      *v5 = a2;
      ++*(_QWORD *)(v3 + 184);
      goto LABEL_3;
    }
    goto LABEL_2;
  }
  v7 = 0;
  while ( a2 != *result )
  {
    if ( *result == -2 )
      v7 = result;
    if ( v5 == ++result )
    {
      if ( !v7 )
        goto LABEL_12;
      *v7 = a2;
      --*(_DWORD *)(v3 + 216);
      ++*(_QWORD *)(v3 + 184);
      *(_WORD *)(a2 + 12) |= 8u;
      return result;
    }
  }
LABEL_3:
  *(_WORD *)(a2 + 12) |= 8u;
  return result;
}
