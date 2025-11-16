// Function: sub_10E26A0
// Address: 0x10e26a0
//
__int64 __fastcall sub_10E26A0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  _BYTE *v4; // rcx
  __int64 v5; // rdi
  __int64 v7; // rax
  _BYTE *v8; // rdx
  _BYTE *v9; // r12
  _BYTE *v10; // rcx
  __int64 v11; // r13
  __int16 v12; // ax
  int v13; // eax
  __int64 v14; // rdx
  _BYTE *v15; // rax
  __int64 v16; // rdx

  if ( a2 )
  {
    v3 = *(_QWORD *)(a2 - 32);
    if ( !v3 || *(_BYTE *)v3 || *(_QWORD *)(v3 + 24) != *(_QWORD *)(a2 + 80) )
      BUG();
    if ( *(_DWORD *)(v3 + 36) == 330 )
    {
      v4 = *(_BYTE **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
      if ( *v4 <= 0x1Cu )
        return 0;
      v5 = *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
      **(_QWORD **)a1 = v4;
      if ( *(_BYTE *)v5 == 17 )
      {
        **(_QWORD **)(a1 + 8) = v5 + 24;
        return 1;
      }
      v14 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v5 + 8) + 8LL) - 17;
      if ( (unsigned int)v14 > 1 )
        return 0;
      if ( *(_BYTE *)v5 > 0x15u )
        return 0;
      v15 = sub_AD7630(v5, *(unsigned __int8 *)(a1 + 16), v14);
      if ( !v15 )
        return 0;
      goto LABEL_23;
    }
  }
  if ( *(_BYTE *)a2 != 86 )
    return 0;
  v7 = *(_QWORD *)(a2 - 96);
  if ( *(_BYTE *)v7 != 82 )
    return 0;
  v8 = *(_BYTE **)(a2 - 64);
  v9 = *(_BYTE **)(v7 - 64);
  v10 = *(_BYTE **)(a2 - 32);
  v11 = *(_QWORD *)(v7 - 32);
  if ( v9 == v8 && (_BYTE *)v11 == v10 )
  {
    v12 = *(_WORD *)(v7 + 2);
LABEL_15:
    v13 = v12 & 0x3F;
    goto LABEL_16;
  }
  if ( (_BYTE *)v11 != v8 || v9 != v10 )
    return 0;
  v12 = *(_WORD *)(v7 + 2);
  if ( v9 == v8 )
    goto LABEL_15;
  v13 = sub_B52870(v12 & 0x3F);
LABEL_16:
  if ( (unsigned int)(v13 - 40) > 1 || *v9 <= 0x1Cu )
    return 0;
  **(_QWORD **)a1 = v9;
  if ( *(_BYTE *)v11 == 17 )
  {
    **(_QWORD **)(a1 + 8) = v11 + 24;
    return 1;
  }
  v16 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v11 + 8) + 8LL) - 17;
  if ( (unsigned int)v16 > 1 )
    return 0;
  if ( *(_BYTE *)v11 > 0x15u )
    return 0;
  v15 = sub_AD7630(v11, *(unsigned __int8 *)(a1 + 16), v16);
  if ( !v15 )
    return 0;
LABEL_23:
  if ( *v15 != 17 )
    return 0;
  **(_QWORD **)(a1 + 8) = v15 + 24;
  return 1;
}
