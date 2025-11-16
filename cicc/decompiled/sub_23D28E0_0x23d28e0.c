// Function: sub_23D28E0
// Address: 0x23d28e0
//
__int64 __fastcall sub_23D28E0(__int64 a1, char *a2)
{
  unsigned __int8 v2; // al
  __int64 v4; // rax
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // r12
  __int64 v13; // rcx
  __int64 v14; // r13
  __int16 v15; // ax
  int v16; // eax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rdx
  _BYTE *v20; // rax
  __int64 v21; // rdx

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
    v7 = *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    v8 = *(_QWORD *)(v7 + 16);
    if ( !v8 )
      return 0;
    if ( *(_QWORD *)(v8 + 8) )
      return 0;
    if ( *(_BYTE *)v7 != 71 )
      return 0;
    v9 = *(_QWORD *)(v7 - 32);
    if ( !v9 )
      return 0;
    v10 = *(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
    **(_QWORD **)a1 = v9;
    if ( *(_BYTE *)v10 == 17 )
    {
      **(_QWORD **)(a1 + 8) = v10 + 24;
      return 1;
    }
    v19 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v10 + 8) + 8LL) - 17;
    if ( (unsigned int)v19 > 1 )
      return 0;
    if ( *(_BYTE *)v10 > 0x15u )
      return 0;
    v20 = sub_AD7630(v10, *(unsigned __int8 *)(a1 + 16), v19);
    if ( !v20 )
      return 0;
    goto LABEL_32;
  }
  if ( v2 != 86 )
    return 0;
  v4 = *((_QWORD *)a2 - 12);
  if ( *(_BYTE *)v4 != 82 )
    return 0;
  v11 = *((_QWORD *)a2 - 8);
  v12 = *(_QWORD *)(v4 - 64);
  v13 = *((_QWORD *)a2 - 4);
  v14 = *(_QWORD *)(v4 - 32);
  if ( v12 == v11 && v14 == v13 )
  {
    v15 = *(_WORD *)(v4 + 2);
LABEL_21:
    v16 = v15 & 0x3F;
    goto LABEL_22;
  }
  if ( v14 != v11 || v12 != v13 )
    return 0;
  v15 = *(_WORD *)(v4 + 2);
  if ( v12 == v11 )
    goto LABEL_21;
  v16 = sub_B52870(v15 & 0x3F);
LABEL_22:
  if ( (unsigned int)(v16 - 38) > 1 )
    return 0;
  v17 = *(_QWORD *)(v12 + 16);
  if ( !v17 )
    return 0;
  if ( *(_QWORD *)(v17 + 8) )
    return 0;
  if ( *(_BYTE *)v12 != 71 )
    return 0;
  v18 = *(_QWORD *)(v12 - 32);
  if ( !v18 )
    return 0;
  **(_QWORD **)a1 = v18;
  if ( *(_BYTE *)v14 == 17 )
  {
    **(_QWORD **)(a1 + 8) = v14 + 24;
    return 1;
  }
  v21 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v14 + 8) + 8LL) - 17;
  if ( (unsigned int)v21 > 1 )
    return 0;
  if ( *(_BYTE *)v14 > 0x15u )
    return 0;
  v20 = sub_AD7630(v14, *(unsigned __int8 *)(a1 + 16), v21);
  if ( !v20 )
    return 0;
LABEL_32:
  if ( *v20 != 17 )
    return 0;
  **(_QWORD **)(a1 + 8) = v20 + 24;
  return 1;
}
