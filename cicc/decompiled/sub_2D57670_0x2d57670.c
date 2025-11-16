// Function: sub_2D57670
// Address: 0x2d57670
//
__int64 __fastcall sub_2D57670(char *a1, _QWORD *a2, __int64 *a3)
{
  char v4; // al
  __int64 v5; // rax
  _BYTE *v7; // rax
  _BYTE *v8; // rax
  _BYTE *v9; // rax
  _BYTE *v10; // r8
  __int64 v11; // rax
  __int64 v12; // rdx
  _BYTE *v13; // rdx
  __int64 v14; // rdx
  _BYTE *v15; // rdx
  char v16; // al

  v4 = *a1;
  if ( *a1 == 42 )
  {
    v7 = (_BYTE *)*((_QWORD *)a1 - 8);
    if ( *v7 <= 0x1Cu )
      return 0;
    *a2 = v7;
    v8 = (_BYTE *)*((_QWORD *)a1 - 4);
    if ( *v8 <= 0x15u )
      goto LABEL_10;
    v4 = *a1;
  }
  if ( v4 != 93 )
  {
    if ( v4 != 44 )
      return 0;
    goto LABEL_12;
  }
  if ( *((_DWORD *)a1 + 20) != 1
    || **((_DWORD **)a1 + 9)
    || (v5 = *((_QWORD *)a1 - 4), *(_BYTE *)v5 != 85)
    || (v14 = *(_QWORD *)(v5 - 32)) == 0
    || *(_BYTE *)v14
    || *(_QWORD *)(v14 + 24) != *(_QWORD *)(v5 + 80)
    || *(_DWORD *)(v14 + 36) != 360
    || (v15 = *(_BYTE **)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF)), *v15 <= 0x1Cu) )
  {
LABEL_6:
    if ( *((_DWORD *)a1 + 20) != 1 )
      return 0;
    if ( **((_DWORD **)a1 + 9) )
      return 0;
    v11 = *((_QWORD *)a1 - 4);
    if ( *(_BYTE *)v11 != 85 )
      return 0;
    v12 = *(_QWORD *)(v11 - 32);
    if ( !v12 )
      return 0;
    if ( *(_BYTE *)v12 )
      return 0;
    if ( *(_QWORD *)(v12 + 24) != *(_QWORD *)(v11 + 80) )
      return 0;
    if ( *(_DWORD *)(v12 + 36) != 372 )
      return 0;
    v13 = *(_BYTE **)(v11 - 32LL * (*(_DWORD *)(v11 + 4) & 0x7FFFFFF));
    if ( *v13 <= 0x1Cu )
      return 0;
    *a2 = v13;
    if ( *(_BYTE *)v11 != 85 )
      return 0;
    v10 = *(_BYTE **)(v11 + 32 * (1LL - (*(_DWORD *)(v11 + 4) & 0x7FFFFFF)));
    if ( *v10 > 0x15u )
      return 0;
    goto LABEL_14;
  }
  *a2 = v15;
  if ( *(_BYTE *)v5 == 85 )
  {
    v8 = *(_BYTE **)(v5 + 32 * (1LL - (*(_DWORD *)(v5 + 4) & 0x7FFFFFF)));
    if ( *v8 <= 0x15u )
    {
LABEL_10:
      *a3 = (__int64)v8;
      return 1;
    }
  }
  v16 = *a1;
  if ( *a1 != 44 )
    goto LABEL_33;
LABEL_12:
  v9 = (_BYTE *)*((_QWORD *)a1 - 8);
  if ( *v9 <= 0x1Cu )
    return 0;
  *a2 = v9;
  v10 = (_BYTE *)*((_QWORD *)a1 - 4);
  if ( *v10 > 0x15u )
  {
    v16 = *a1;
LABEL_33:
    if ( v16 == 93 )
      goto LABEL_6;
    return 0;
  }
LABEL_14:
  *a3 = (__int64)v10;
  *a3 = sub_AD6890((__int64)v10, 0);
  return 1;
}
