// Function: sub_15ABCE0
// Address: 0x15abce0
//
_QWORD *__fastcall sub_15ABCE0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *result; // rax
  __int64 v4; // rbx
  char v6; // dl
  _QWORD *v7; // rsi
  unsigned int v8; // edi
  _QWORD *v9; // rcx

  result = *(_QWORD **)(a3 + 24 * (1LL - (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)));
  v4 = result[3];
  if ( !v4 || *(_BYTE *)v4 != 25 )
    return result;
  result = *(_QWORD **)(a1 + 408);
  if ( *(_QWORD **)(a1 + 416) == result )
  {
    v7 = &result[*(unsigned int *)(a1 + 428)];
    v8 = *(_DWORD *)(a1 + 428);
    if ( result != v7 )
    {
      v9 = 0;
      while ( v4 != *result )
      {
        if ( *result == -2 )
          v9 = result;
        if ( v7 == ++result )
        {
          if ( !v9 )
            goto LABEL_15;
          *v9 = v4;
          --*(_DWORD *)(a1 + 432);
          ++*(_QWORD *)(a1 + 400);
          goto LABEL_5;
        }
      }
      return result;
    }
LABEL_15:
    if ( v8 < *(_DWORD *)(a1 + 424) )
    {
      *(_DWORD *)(a1 + 428) = v8 + 1;
      *v7 = v4;
      ++*(_QWORD *)(a1 + 400);
      goto LABEL_5;
    }
  }
  result = (_QWORD *)sub_16CCBA0(a1 + 400, v4);
  if ( v6 )
  {
LABEL_5:
    sub_15AB790(a1, *(unsigned __int8 **)(v4 - 8LL * *(unsigned int *)(v4 + 8)));
    return (_QWORD *)sub_15ABBA0(a1, *(unsigned __int8 **)(v4 + 8 * (3LL - *(unsigned int *)(v4 + 8))));
  }
  return result;
}
