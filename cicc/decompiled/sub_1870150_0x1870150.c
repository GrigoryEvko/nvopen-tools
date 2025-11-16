// Function: sub_1870150
// Address: 0x1870150
//
__int64 __fastcall sub_1870150(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  unsigned __int64 v6; // rdx
  _QWORD *v7; // rax
  _QWORD *v8; // rdi
  __int64 v9; // rsi
  __int64 v10; // rcx

  v5 = sub_15E4F10(a2);
  if ( v5 )
  {
    v6 = v5;
    v7 = *(_QWORD **)(a3 + 16);
    if ( !v7 )
      goto LABEL_12;
    v8 = (_QWORD *)(a3 + 8);
    do
    {
      while ( 1 )
      {
        v9 = v7[2];
        v10 = v7[3];
        if ( v7[4] >= v6 )
          break;
        v7 = (_QWORD *)v7[3];
        if ( !v10 )
          goto LABEL_7;
      }
      v8 = v7;
      v7 = (_QWORD *)v7[2];
    }
    while ( v9 );
LABEL_7:
    if ( (_QWORD *)(a3 + 8) == v8 || v8[4] > v6 )
    {
LABEL_12:
      if ( *(_BYTE *)(a2 + 16) == 3 || !*(_BYTE *)(a2 + 16) )
        *(_QWORD *)(a2 + 48) = 0;
      if ( (*(_BYTE *)(a2 + 32) & 0xFu) - 7 > 1 )
        goto LABEL_13;
    }
  }
  else if ( (*(_BYTE *)(a2 + 32) & 0xFu) - 7 > 1 && !sub_1870060(a1, a2) )
  {
LABEL_13:
    *(_WORD *)(a2 + 32) = *(_WORD *)(a2 + 32) & 0xBFC0 | 0x4007;
    return 1;
  }
  return 0;
}
