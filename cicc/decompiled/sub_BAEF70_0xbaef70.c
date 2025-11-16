// Function: sub_BAEF70
// Address: 0xbaef70
//
__int64 __fastcall sub_BAEF70(__int64 a1, unsigned __int64 a2)
{
  _QWORD *v4; // rax
  _QWORD *v5; // rdx
  __int64 v6; // rsi
  __int64 v7; // rcx
  __int64 result; // rax
  unsigned __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rdx

  v4 = *(_QWORD **)(a1 + 16);
  if ( !v4 )
    return 1;
  v5 = (_QWORD *)(a1 + 8);
  do
  {
    while ( 1 )
    {
      v6 = v4[2];
      v7 = v4[3];
      if ( a2 <= v4[4] )
        break;
      v4 = (_QWORD *)v4[3];
      if ( !v7 )
        goto LABEL_6;
    }
    v5 = v4;
    v4 = (_QWORD *)v4[2];
  }
  while ( v6 );
LABEL_6:
  result = 1;
  if ( (_QWORD *)(a1 + 8) != v5 && a2 >= v5[4] )
  {
    v9 = *(_BYTE *)(a1 + 343) & 0xF8 | (unsigned __int64)(v5 + 4) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v9 )
    {
      v10 = *(_QWORD *)(v9 + 24);
      v11 = *(_QWORD *)(v9 + 32);
      if ( v10 != v11 )
      {
        result = *(unsigned __int8 *)(a1 + 336);
        while ( (_BYTE)result )
        {
          if ( *(char *)(*(_QWORD *)v10 + 12LL) < 0 )
            return result;
          v10 += 8;
          if ( v11 == v10 )
            return 0;
        }
        return 1;
      }
    }
  }
  return result;
}
