// Function: sub_1AED280
// Address: 0x1aed280
//
char __fastcall sub_1AED280(__int64 a1, unsigned int a2)
{
  __int64 v2; // rdx
  __int64 v3; // r12
  __int64 v4; // rbx
  _BYTE *v5; // rdx
  unsigned __int8 v6; // cl
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rax
  __int64 v10; // r14
  __int64 *v11; // r14
  __int64 *v12; // r15
  __int64 v13; // r13
  __int64 v14; // rdx
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rax
  char v17; // dl
  bool v18; // zf
  unsigned __int64 v19; // r12
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rbx
  __int64 v23; // rdx
  __int64 *v24; // rbx
  char v25; // al
  __int64 v26; // rax
  unsigned __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rdx
  unsigned int v31; // [rsp+Ch] [rbp-34h]

  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v2 = *(_QWORD *)(a1 - 8);
  else
    v2 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  v3 = a2;
  v4 = 3LL * a2;
  v5 = *(_BYTE **)(v2 + v4 * 8);
  if ( *(_BYTE *)(*(_QWORD *)v5 + 8LL) == 8 )
    goto LABEL_14;
  if ( v5[16] > 0x10u )
    goto LABEL_7;
  v6 = *(_BYTE *)(a1 + 16);
  switch ( v6 )
  {
    case 0x1Bu:
    case 0x56u:
      LOBYTE(v7) = a2 == 0;
      return v7;
    case 0x1Du:
    case 0x4Eu:
      if ( v6 <= 0x17u )
      {
        if ( *(_BYTE *)(MEMORY[0xFFFFFFFFFFFFFFB8] + 16LL) == 20 )
          goto LABEL_14;
        v19 = 0;
        goto LABEL_34;
      }
      v8 = a1 | 4;
      if ( v6 == 78 )
        goto LABEL_12;
      v9 = 0;
      if ( v6 != 29 )
        goto LABEL_31;
      v8 = a1 & 0xFFFFFFFFFFFFFFFBLL;
LABEL_12:
      v9 = v8 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v8 & 4) != 0 )
      {
        if ( *(_BYTE *)(*(_QWORD *)(v9 - 24) + 16LL) == 20 )
        {
LABEL_14:
          LOBYTE(v7) = 0;
          return v7;
        }
      }
      else
      {
LABEL_31:
        if ( *(_BYTE *)(*(_QWORD *)(v9 - 72) + 16LL) == 20 )
          goto LABEL_14;
      }
      if ( v6 == 78 )
      {
        v26 = *(_QWORD *)(a1 - 24);
        if ( !*(_BYTE *)(v26 + 16) && (*(_BYTE *)(v26 + 33) & 0x20) != 0 )
          goto LABEL_14;
        v27 = a1 | 4;
      }
      else
      {
        v19 = 0;
        if ( v6 != 29 )
          goto LABEL_34;
        v27 = a1 & 0xFFFFFFFFFFFFFFFBLL;
      }
      v19 = v27 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v27 & 4) == 0 )
      {
LABEL_34:
        v31 = a2;
        if ( *(char *)(v19 + 23) >= 0 )
          goto LABEL_7;
        v20 = sub_1648A40(v19);
        v22 = v20 + v21;
        if ( *(char *)(v19 + 23) < 0 )
        {
          if ( (unsigned int)((v22 - sub_1648A40(v19)) >> 4) )
          {
            if ( *(char *)(v19 + 23) < 0 )
              goto LABEL_38;
LABEL_65:
            BUG();
          }
LABEL_7:
          LOBYTE(v7) = 1;
          return v7;
        }
LABEL_66:
        if ( !(unsigned int)(v22 >> 4) )
          goto LABEL_7;
        goto LABEL_65;
      }
      v31 = a2;
      if ( *(char *)((v27 & 0xFFFFFFFFFFFFFFF8LL) + 23) >= 0 )
        goto LABEL_7;
      v28 = sub_1648A40(v19);
      v22 = v28 + v29;
      if ( *(char *)(v19 + 23) >= 0 )
        goto LABEL_66;
      if ( !(unsigned int)((v22 - sub_1648A40(v19)) >> 4) )
        goto LABEL_7;
      if ( *(char *)(v19 + 23) >= 0 )
        goto LABEL_65;
LABEL_38:
      if ( v31 < *(_DWORD *)(sub_1648A40(v19) + 8) )
        goto LABEL_7;
      if ( *(char *)(v19 + 23) >= 0 )
        BUG();
      v7 = sub_1648A40(v19);
      LOBYTE(v7) = v31 >= *(_DWORD *)(v7 + v23 - 4);
      return v7;
    case 0x35u:
      LODWORD(v7) = sub_15F8F00(a1) ^ 1;
      return v7;
    case 0x38u:
      if ( !a2 )
        goto LABEL_7;
      v10 = (*(_BYTE *)(a1 + 23) & 0x40) != 0 ? *(_QWORD *)(a1 - 8) : a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      v11 = (__int64 *)(v10 + 24);
      v12 = v11;
      v13 = sub_16348C0(a1) | 4;
      v14 = v13;
      do
      {
        v15 = v14 & 0xFFFFFFFFFFFFFFF8LL;
        v16 = v14 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v14 & 4) == 0 || !v15 )
          v16 = sub_1643D30(v15, *v12);
        v17 = *(_BYTE *)(v16 + 8);
        if ( ((v17 - 14) & 0xFD) != 0 )
        {
          v18 = v17 == 13;
          v14 = 0;
          if ( v18 )
            v14 = v16;
        }
        else
        {
          v14 = *(_QWORD *)(v16 + 24) | 4LL;
        }
        v12 += 3;
        --v3;
      }
      while ( v3 );
      v24 = &v11[v4];
      if ( v11 == v24 )
        goto LABEL_7;
      while ( (v13 & 4) != 0 )
      {
        v13 &= 0xFFFFFFFFFFFFFFF8LL;
        if ( !v13 )
          v13 = sub_1643D30(0, *v11);
        v25 = *(_BYTE *)(v13 + 8);
        if ( ((v25 - 14) & 0xFD) != 0 )
        {
          if ( v25 != 13 )
            v13 = 0;
        }
        else
        {
          v13 = *(_QWORD *)(v13 + 24) | 4LL;
        }
        v11 += 3;
        if ( v24 == v11 )
          goto LABEL_7;
      }
      goto LABEL_14;
    case 0x55u:
      LOBYTE(v7) = a2 != 2;
      return v7;
    case 0x57u:
      LOBYTE(v7) = a2 <= 1;
      return v7;
    default:
      goto LABEL_7;
  }
}
