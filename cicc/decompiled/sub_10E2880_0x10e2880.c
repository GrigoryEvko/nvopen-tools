// Function: sub_10E2880
// Address: 0x10e2880
//
__int64 __fastcall sub_10E2880(__int64 a1, char *a2)
{
  unsigned __int8 v2; // al
  __int64 v4; // rax
  __int64 v6; // rax
  _BYTE *v7; // rcx
  __int64 v8; // rdi
  _BYTE *v9; // rdx
  _BYTE *v10; // r12
  _BYTE *v11; // rcx
  __int64 v12; // r13
  __int16 v13; // ax
  int v14; // eax
  __int64 v15; // rdx
  _BYTE *v16; // rax
  __int64 v17; // rdx

  v2 = *a2;
  if ( (unsigned __int8)*a2 <= 0x1Cu )
    return 0;
  if ( v2 == 85 )
  {
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
    v7 = *(_BYTE **)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    if ( (unsigned __int8)(*v7 - 42) > 0x11u )
      return 0;
    v8 = *(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
    **(_QWORD **)a1 = v7;
    if ( *(_BYTE *)v8 == 17 )
    {
      **(_QWORD **)(a1 + 8) = v8 + 24;
      return 1;
    }
    v15 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v8 + 8) + 8LL) - 17;
    if ( (unsigned int)v15 > 1 )
      return 0;
    if ( *(_BYTE *)v8 > 0x15u )
      return 0;
    v16 = sub_AD7630(v8, *(unsigned __int8 *)(a1 + 16), v15);
    if ( !v16 )
      return 0;
    goto LABEL_30;
  }
  if ( v2 != 86 )
    return 0;
  v4 = *((_QWORD *)a2 - 12);
  if ( *(_BYTE *)v4 != 82 )
    return 0;
  v9 = (_BYTE *)*((_QWORD *)a2 - 8);
  v10 = *(_BYTE **)(v4 - 64);
  v11 = (_BYTE *)*((_QWORD *)a2 - 4);
  v12 = *(_QWORD *)(v4 - 32);
  if ( v10 == v9 && (_BYTE *)v12 == v11 )
  {
    v13 = *(_WORD *)(v4 + 2);
LABEL_18:
    v14 = v13 & 0x3F;
    goto LABEL_19;
  }
  if ( (_BYTE *)v12 != v9 || v10 != v11 )
    return 0;
  v13 = *(_WORD *)(v4 + 2);
  if ( v10 == v9 )
    goto LABEL_18;
  v14 = sub_B52870(v13 & 0x3F);
LABEL_19:
  if ( (unsigned int)(v14 - 38) > 1 || (unsigned __int8)(*v10 - 42) > 0x11u )
    return 0;
  **(_QWORD **)a1 = v10;
  if ( *(_BYTE *)v12 == 17 )
  {
    **(_QWORD **)(a1 + 8) = v12 + 24;
    return 1;
  }
  v17 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v12 + 8) + 8LL) - 17;
  if ( (unsigned int)v17 > 1 )
    return 0;
  if ( *(_BYTE *)v12 > 0x15u )
    return 0;
  v16 = sub_AD7630(v12, *(unsigned __int8 *)(a1 + 16), v17);
  if ( !v16 )
    return 0;
LABEL_30:
  if ( *v16 != 17 )
    return 0;
  **(_QWORD **)(a1 + 8) = v16 + 24;
  return 1;
}
