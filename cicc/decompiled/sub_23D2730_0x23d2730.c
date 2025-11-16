// Function: sub_23D2730
// Address: 0x23d2730
//
__int64 __fastcall sub_23D2730(__int64 a1, char *a2)
{
  unsigned __int8 v2; // al
  __int64 v3; // rax
  unsigned int v4; // r12d
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r13
  char *v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // r12
  __int64 v12; // rcx
  __int16 v13; // ax
  int v14; // eax
  __int64 v15; // rax
  __int64 v16; // rdx
  _BYTE *v17; // rax

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
    v7 = *(_QWORD *)(*(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)] + 16LL);
    if ( !v7 || *(_QWORD *)(v7 + 8) )
      return 0;
    v8 = *(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
    v9 = *(char **)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    goto LABEL_23;
  }
  if ( v2 != 86 )
    return 0;
  v3 = *((_QWORD *)a2 - 12);
  if ( *(_BYTE *)v3 != 82 )
    return 0;
  v10 = *((_QWORD *)a2 - 8);
  v11 = *(_QWORD *)(v3 - 64);
  v12 = *((_QWORD *)a2 - 4);
  v8 = *(_QWORD *)(v3 - 32);
  if ( v11 == v10 && v8 == v12 )
  {
    v13 = *(_WORD *)(v3 + 2);
LABEL_18:
    v14 = v13 & 0x3F;
    goto LABEL_19;
  }
  if ( v8 != v10 || v11 != v12 )
    return 0;
  v13 = *(_WORD *)(v3 + 2);
  if ( v11 == v10 )
    goto LABEL_18;
  v14 = sub_B52870(v13 & 0x3F);
LABEL_19:
  if ( (unsigned int)(v14 - 38) > 1 )
    return 0;
  v15 = *(_QWORD *)(v11 + 16);
  if ( !v15 || *(_QWORD *)(v15 + 8) )
    return 0;
  v9 = (char *)v11;
LABEL_23:
  v4 = sub_23D2510(a1, v9);
  if ( (_BYTE)v4 )
  {
    if ( *(_BYTE *)v8 == 17 )
    {
      **(_QWORD **)(a1 + 24) = v8 + 24;
      return v4;
    }
    v16 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v8 + 8) + 8LL) - 17;
    if ( (unsigned int)v16 <= 1 && *(_BYTE *)v8 <= 0x15u )
    {
      v17 = sub_AD7630(v8, *(unsigned __int8 *)(a1 + 32), v16);
      if ( v17 )
      {
        if ( *v17 == 17 )
        {
          **(_QWORD **)(a1 + 24) = v17 + 24;
          return v4;
        }
      }
    }
  }
  return 0;
}
