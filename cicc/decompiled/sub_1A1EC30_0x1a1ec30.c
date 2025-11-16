// Function: sub_1A1EC30
// Address: 0x1a1ec30
//
char __fastcall sub_1A1EC30(unsigned __int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4, _BYTE *a5)
{
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // rbx
  _QWORD *v10; // rax
  unsigned __int8 v11; // cl
  _QWORD *v12; // rdx
  unsigned __int64 v13; // r8
  unsigned __int64 v14; // rbx
  unsigned __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // rsi
  char v18; // al
  unsigned int v19; // ebx
  int v20; // esi
  __int64 v21; // rbx
  __int64 v22; // rdi
  unsigned int v23; // r12d
  int v24; // eax
  char v25; // al
  __int64 v26; // rdx
  __int64 *v28; // [rsp+0h] [rbp-50h]
  __int64 v29; // [rsp+0h] [rbp-50h]
  unsigned __int64 v30; // [rsp+8h] [rbp-48h]
  _QWORD *v32; // [rsp+18h] [rbp-38h]

  v7 = sub_127FA20(a4, a3);
  v8 = a1[2];
  v9 = v7;
  v10 = sub_1648700(v8 & 0xFFFFFFFFFFFFFFF8LL);
  v11 = *((_BYTE *)v10 + 16);
  if ( v11 <= 0x17u )
    goto LABEL_7;
  v12 = v10;
  v13 = a1[1] - a2;
  v14 = (unsigned __int64)(v9 + 7) >> 3;
  if ( v11 == 78 )
  {
    v16 = *(v10 - 3);
    if ( *(_BYTE *)(v16 + 16) || (*(_BYTE *)(v16 + 33) & 0x20) == 0 )
      goto LABEL_7;
    v20 = *(_DWORD *)(v16 + 36);
    LOBYTE(v16) = 1;
    if ( (unsigned int)(v20 - 116) <= 1 )
      return v16;
    if ( v13 > v14 )
      goto LABEL_7;
    if ( (unsigned int)(v20 - 133) > 4 || ((1LL << ((unsigned __int8)v20 + 123)) & 0x15) == 0 )
    {
      LOBYTE(v16) = (unsigned int)(v20 - 116) <= 1;
      return v16;
    }
    v21 = *((_DWORD *)v12 + 5) & 0xFFFFFFF;
    v22 = v12[3 * (3 - v21)];
    v23 = *(_DWORD *)(v22 + 32);
    if ( v23 <= 0x40 )
    {
      if ( *(_QWORD *)(v22 + 24) )
        goto LABEL_7;
    }
    else
    {
      v32 = v12;
      v24 = sub_16A57B0(v22 + 24);
      v12 = v32;
      if ( v23 != v24 )
        goto LABEL_7;
    }
    if ( *(_BYTE *)(v12[3 * (2 - v21)] + 16LL) <= 0x10u )
      return (v8 >> 2) & 1;
LABEL_7:
    LOBYTE(v16) = 0;
    return v16;
  }
  if ( v13 > v14 )
    goto LABEL_7;
  v15 = *a1 - a2;
  v30 = a1[1] - a2;
  if ( v11 == 54 )
  {
    if ( (*((_BYTE *)v10 + 18) & 1) != 0 )
      goto LABEL_7;
    v28 = v10;
    if ( (unsigned __int64)(sub_127FA20(a4, *v10) + 7) >> 3 > v14 || a2 > *a1 )
      goto LABEL_7;
    v17 = *v28;
    v18 = *(_BYTE *)(*v28 + 8);
    if ( v18 != 16 )
    {
      if ( !v15 && v30 == v14 )
      {
        *a5 = 1;
        v17 = *v28;
        if ( *(_BYTE *)(*v28 + 8) == 11 )
          goto LABEL_16;
LABEL_41:
        v26 = v17;
        v17 = a3;
        goto LABEL_36;
      }
      if ( v18 == 11 )
        goto LABEL_16;
    }
    if ( v15 || v30 != v14 )
      goto LABEL_7;
    goto LABEL_41;
  }
  if ( v11 != 55 )
    goto LABEL_7;
  if ( (*((_BYTE *)v10 + 18) & 1) != 0 )
    goto LABEL_7;
  v29 = *(_QWORD *)*(v10 - 6);
  if ( v14 < (unsigned __int64)(sub_127FA20(a4, v29) + 7) >> 3 || a2 > *a1 )
    goto LABEL_7;
  v17 = v29;
  v25 = *(_BYTE *)(v29 + 8);
  if ( v25 == 16 )
    goto LABEL_44;
  if ( !v15 && v30 == v14 )
  {
    *a5 = 1;
    if ( *(_BYTE *)(v29 + 8) != 11 )
    {
LABEL_35:
      v26 = a3;
LABEL_36:
      LOBYTE(v16) = sub_1A1E350(a4, v17, v26);
      return v16;
    }
    goto LABEL_16;
  }
  if ( v25 != 11 )
  {
LABEL_44:
    if ( !v15 && v30 == v14 )
      goto LABEL_35;
    goto LABEL_7;
  }
LABEL_16:
  v19 = *(_DWORD *)(v17 + 8);
  LOBYTE(v16) = v19 >> 8 >= ((sub_127FA20(a4, v17) + 7) & 0xFFFFFFFFFFFFFFF8LL);
  return v16;
}
