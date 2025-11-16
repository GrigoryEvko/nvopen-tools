// Function: sub_100AFE0
// Address: 0x100afe0
//
__int64 __fastcall sub_100AFE0(unsigned __int8 ***a1, __int64 a2)
{
  __int64 v3; // rdi
  unsigned int v4; // eax
  unsigned int v5; // r12d
  __int64 v6; // r14
  _BYTE *v7; // rdi
  unsigned __int8 *v8; // r15
  unsigned int v9; // eax
  __int64 *v11; // rbx
  unsigned __int8 *v12; // rbx
  char v13; // al
  __int64 v14; // rsi
  char v15; // al
  __int64 v16; // rsi
  unsigned __int8 *v17; // rax
  unsigned __int8 *v18; // rax
  char v19; // al
  __int64 v20; // rsi
  unsigned __int8 *v21; // rax

  if ( *(_BYTE *)a2 <= 0x1Cu )
    return 0;
  v3 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 <= 1 )
    v3 = **(_QWORD **)(v3 + 16);
  LOBYTE(v4) = sub_BCAC40(v3, 1);
  v5 = v4;
  if ( !(_BYTE)v4 )
    return 0;
  if ( *(_BYTE *)a2 != 57 )
  {
    if ( *(_BYTE *)a2 != 86 )
      return 0;
    v6 = *(_QWORD *)(a2 - 96);
    if ( *(_QWORD *)(a2 + 8) != *(_QWORD *)(v6 + 8) )
      return 0;
    v7 = *(_BYTE **)(a2 - 32);
    if ( *v7 > 0x15u )
      return 0;
    v8 = *(unsigned __int8 **)(a2 - 64);
    LOBYTE(v9) = sub_AC30F0((__int64)v7);
    v5 = v9;
    if ( !(_BYTE)v9 )
      return 0;
    **a1 = (unsigned __int8 *)v6;
    if ( *(_BYTE *)v6 != 59 )
    {
LABEL_25:
      if ( v8 )
      {
        **a1 = v8;
        v5 = sub_996420(a1 + 1, 30, v8);
        if ( (_BYTE)v5 )
          goto LABEL_22;
      }
      return 0;
    }
    v19 = sub_995B10(a1 + 1, *(_QWORD *)(v6 - 64));
    v20 = *(_QWORD *)(v6 - 32);
    if ( v19 && v20 )
    {
      *a1[2] = (unsigned __int8 *)v20;
    }
    else
    {
      if ( !(unsigned __int8)sub_995B10(a1 + 1, v20) )
        goto LABEL_25;
      v21 = *(unsigned __int8 **)(v6 - 64);
      if ( !v21 )
        goto LABEL_25;
      *a1[2] = v21;
    }
    if ( v8 )
    {
      *a1[3] = v8;
      return v5;
    }
    return 0;
  }
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v11 = *(__int64 **)(a2 - 8);
  else
    v11 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v6 = *v11;
  v12 = (unsigned __int8 *)v11[4];
  if ( v6 )
  {
    **a1 = (unsigned __int8 *)v6;
    if ( *(_BYTE *)v6 == 59 )
    {
      v15 = sub_995B10(a1 + 1, *(_QWORD *)(v6 - 64));
      v16 = *(_QWORD *)(v6 - 32);
      if ( v15 && v16 )
      {
        *a1[2] = (unsigned __int8 *)v16;
      }
      else
      {
        if ( !(unsigned __int8)sub_995B10(a1 + 1, v16) )
          goto LABEL_16;
        v17 = *(unsigned __int8 **)(v6 - 64);
        if ( !v17 )
          goto LABEL_16;
        *a1[2] = v17;
      }
      if ( v12 )
      {
        *a1[3] = v12;
        return v5;
      }
      return 0;
    }
  }
LABEL_16:
  if ( !v12 )
    return 0;
  **a1 = v12;
  if ( *v12 != 59 )
    return 0;
  v13 = sub_995B10(a1 + 1, *((_QWORD *)v12 - 8));
  v14 = *((_QWORD *)v12 - 4);
  if ( v13 && v14 )
  {
    *a1[2] = (unsigned __int8 *)v14;
  }
  else
  {
    if ( !(unsigned __int8)sub_995B10(a1 + 1, v14) )
      return 0;
    v18 = (unsigned __int8 *)*((_QWORD *)v12 - 8);
    if ( !v18 )
      return 0;
    *a1[2] = v18;
  }
  if ( v6 )
  {
LABEL_22:
    *a1[3] = (unsigned __int8 *)v6;
    return v5;
  }
  return 0;
}
