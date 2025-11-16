// Function: sub_10ECBA0
// Address: 0x10ecba0
//
__int64 __fastcall sub_10ECBA0(_QWORD **a1, char *a2)
{
  unsigned __int8 v2; // al
  __int64 v4; // rax
  __int64 v6; // rax
  __int64 v7; // rcx
  _BYTE *v8; // rdi
  _BYTE *v9; // rdx
  _BYTE *v10; // r12
  _BYTE *v11; // rcx
  _BYTE *v12; // r13
  __int16 v13; // ax
  int v14; // eax

  v2 = *a2;
  if ( (unsigned __int8)*a2 <= 0x1Cu )
    return 0;
  if ( v2 != 85 )
  {
    if ( v2 != 86 )
      return 0;
    v4 = *((_QWORD *)a2 - 12);
    if ( *(_BYTE *)v4 != 82 )
      return 0;
    v9 = (_BYTE *)*((_QWORD *)a2 - 8);
    v10 = *(_BYTE **)(v4 - 64);
    v11 = (_BYTE *)*((_QWORD *)a2 - 4);
    v12 = *(_BYTE **)(v4 - 32);
    if ( v10 == v9 && v12 == v11 )
    {
      v13 = *(_WORD *)(v4 + 2);
    }
    else
    {
      if ( v12 != v9 || v10 != v11 )
        return 0;
      v13 = *(_WORD *)(v4 + 2);
      if ( v10 != v9 )
      {
        v14 = sub_B52870(v13 & 0x3F);
        goto LABEL_22;
      }
    }
    v14 = v13 & 0x3F;
LABEL_22:
    if ( (unsigned int)(v14 - 38) > 1 )
      return 0;
    if ( !v10 )
      return 0;
    **a1 = v10;
    if ( *v12 > 0x15u )
      return 0;
    *a1[1] = v12;
    if ( *v12 <= 0x15u )
    {
      if ( *v12 != 5 )
      {
        v8 = v12;
        return (unsigned int)sub_AD6CA0((__int64)v8) ^ 1;
      }
      return 0;
    }
    return 1;
  }
  v6 = *((_QWORD *)a2 - 4);
  if ( !v6 )
    return 0;
  if ( *(_BYTE *)v6 )
    return 0;
  if ( *(_QWORD *)(v6 + 24) != *((_QWORD *)a2 + 10) )
    return 0;
  if ( (*(_BYTE *)(v6 + 33) & 0x20) == 0 )
    return 0;
  if ( *(_DWORD *)(v6 + 36) != 329 )
    return 0;
  v7 = *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  if ( !v7 )
    return 0;
  v8 = *(_BYTE **)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
  **a1 = v7;
  if ( *v8 > 0x15u )
    return 0;
  *a1[1] = v8;
  if ( *v8 > 0x15u )
    return 1;
  if ( *v8 != 5 )
    return (unsigned int)sub_AD6CA0((__int64)v8) ^ 1;
  return 0;
}
