// Function: sub_1D471D0
// Address: 0x1d471d0
//
_QWORD *__fastcall sub_1D471D0(__int64 a1)
{
  _QWORD *v2; // rax
  _QWORD *v3; // rdi
  _QWORD *v4; // rdx
  unsigned __int64 v5; // rcx
  __int64 v6; // rax
  __int16 v7; // ax
  unsigned __int64 v8; // rsi
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  __int64 v11; // rsi
  unsigned __int64 v13; // rax

  v2 = (_QWORD *)sub_1DD5EE0(a1);
  v3 = *(_QWORD **)(a1 + 32);
  if ( v3 == v2 )
    return *(_QWORD **)(a1 + 32);
  v4 = v2;
  v5 = *v2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v5 )
    BUG();
  v6 = *(_QWORD *)v5;
  if ( (*(_QWORD *)v5 & 4) == 0 && (*(_BYTE *)(v5 + 46) & 4) != 0 )
  {
    while ( 1 )
    {
      v13 = v6 & 0xFFFFFFFFFFFFFFF8LL;
      v5 = v13;
      if ( (*(_BYTE *)(v13 + 46) & 4) == 0 )
        break;
      v6 = *(_QWORD *)v13;
    }
  }
  v7 = **(_WORD **)(v5 + 16);
  if ( v7 == 9 )
    goto LABEL_15;
LABEL_5:
  if ( v7 == 15 )
  {
LABEL_15:
    while ( 1 )
    {
      v11 = *(_QWORD *)(v5 + 32);
      if ( *(_BYTE *)v11
        || (*(_BYTE *)(v11 + 3) & 0x10) == 0
        || v7 != 9 && (*(_BYTE *)(v11 + 40) || *(int *)(v11 + 8) <= 0 && *(int *)(v11 + 48) > 0) )
      {
        break;
      }
LABEL_7:
      if ( v3 == (_QWORD *)v5 )
        return *(_QWORD **)(a1 + 32);
      v8 = *(_QWORD *)v5 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v8 )
        BUG();
      v9 = *(_QWORD *)v8;
      v10 = *(_QWORD *)v5 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_QWORD *)v8 & 4) == 0 && (*(_BYTE *)(v8 + 46) & 4) != 0 )
      {
        while ( 1 )
        {
          v10 = v9 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)((v9 & 0xFFFFFFFFFFFFFFF8LL) + 46) & 4) == 0 )
            break;
          v9 = *(_QWORD *)v10;
        }
      }
      v4 = (_QWORD *)v5;
      v5 = v10;
      v7 = **(_WORD **)(v10 + 16);
      if ( v7 != 9 )
        goto LABEL_5;
    }
  }
  else if ( v7 == 12 )
  {
    goto LABEL_7;
  }
  return v4;
}
