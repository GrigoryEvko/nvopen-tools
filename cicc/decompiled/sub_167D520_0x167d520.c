// Function: sub_167D520
// Address: 0x167d520
//
__int64 __fastcall sub_167D520(__int64 a1, char *a2)
{
  int v3; // eax
  __int64 v4; // r13
  const char *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  int v8; // edx
  _BYTE *v9; // r14
  int v10; // eax
  char v11; // dl
  int v12; // eax
  unsigned int v13; // eax
  unsigned int v14; // r13d
  unsigned __int64 v15; // r15
  _QWORD *v16; // rax
  _QWORD *v17; // r8
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rax
  _QWORD *v21; // rax
  __int64 v22; // rdx
  _BOOL8 v23; // rdi
  unsigned int v24; // eax
  unsigned __int8 v26; // al
  unsigned __int8 v27; // dl
  unsigned __int8 v28; // cl
  int v29; // edx
  int v30; // eax
  char v31; // al
  char v32; // al
  char v33; // al
  int v34; // eax
  int v35; // edx
  __int64 v36; // rdi
  unsigned int v37; // r13d
  __int64 v38; // [rsp+8h] [rbp-58h]
  _QWORD *v39; // [rsp+10h] [rbp-50h]
  __int64 v40; // [rsp+18h] [rbp-48h]
  _QWORD *v41; // [rsp+18h] [rbp-48h]
  char v42; // [rsp+27h] [rbp-39h] BYREF
  char *v43; // [rsp+28h] [rbp-38h] BYREF

  if ( (a2[23] & 0x20) == 0 )
  {
    v8 = *(_DWORD *)(a1 + 72);
    if ( (v8 & 2) != 0 )
      goto LABEL_7;
LABEL_35:
    if ( (v8 & 1) == 0 )
    {
      v10 = a2[32] & 0xF;
      v11 = a2[32] & 0xF;
      if ( (unsigned int)(v10 - 7) <= 1 )
        return 0;
LABEL_37:
      if ( (unsigned int)(v10 - 2) <= 1 || v11 == 1 )
        return 0;
    }
LABEL_12:
    v9 = 0;
    goto LABEL_13;
  }
  v3 = a2[32] & 0xF;
  if ( v3 == 7 || v3 == 8 )
  {
    v12 = *(_DWORD *)(a1 + 72);
    if ( (v12 & 2) != 0 || (v12 & 1) == 0 )
      return 0;
    goto LABEL_12;
  }
  v4 = **(_QWORD **)a1;
  v5 = sub_1649960((__int64)a2);
  v7 = sub_1632000(v4, (__int64)v5, v6);
  v8 = *(_DWORD *)(a1 + 72);
  v9 = (_BYTE *)v7;
  if ( !v7 || (*(_BYTE *)(v7 + 32) & 0xFu) - 7 <= 1 )
  {
    if ( (v8 & 2) != 0 )
    {
LABEL_7:
      if ( (a2[32] & 0xF) != 6 )
        return 0;
      if ( (v8 & 1) != 0 )
        goto LABEL_12;
      v10 = a2[32] & 0xF;
      v11 = a2[32] & 0xF;
      goto LABEL_37;
    }
    goto LABEL_35;
  }
  if ( (v8 & 2) != 0 )
  {
    if ( (a2[32] & 0xF) == 6 )
      goto LABEL_13;
    if ( !sub_15E4F60(v7) )
      return 0;
  }
  v26 = a2[32];
  if ( ((v26 + 10) & 0xFu) > 2 )
  {
    if ( v9[16] == 3 && a2[16] == 3 )
    {
      if ( sub_15E4F60((__int64)v9) && sub_15E4F60((__int64)a2) && ((v9[80] & 1) == 0 || (a2[80] & 1) == 0) )
      {
        v9[80] &= ~1u;
        a2[80] &= ~1u;
      }
      v27 = v9[32];
      v26 = a2[32];
      v28 = v27 & 0xF;
      if ( (v27 & 0xF) == 0xA )
      {
        if ( (v26 & 0xF) != 0xA )
        {
          v34 = (v26 >> 4) & 3;
          v35 = (v27 >> 4) & 3;
          if ( v34 == 1 || v35 == 1 )
          {
            v9[32] = v9[32] & 0xCF | 0x10;
            v31 = 1;
            goto LABEL_70;
          }
          if ( v35 == 2 || v34 == 2 )
          {
            v9[32] = v9[32] & 0xCF | 0x20;
            v31 = 2;
            goto LABEL_70;
          }
          goto LABEL_52;
        }
        v37 = (unsigned int)(1 << (*((_DWORD *)a2 + 8) >> 15)) >> 1;
        if ( (unsigned int)(1 << (*((_DWORD *)v9 + 8) >> 15)) >> 1 >= v37 )
          v37 = (unsigned int)(1 << (*((_DWORD *)v9 + 8) >> 15)) >> 1;
        sub_15E4CC0((__int64)a2, v37);
        sub_15E4CC0((__int64)v9, v37);
        v27 = v9[32];
        v26 = a2[32];
        v28 = v27 & 0xF;
      }
    }
    else
    {
      v27 = v9[32];
      v28 = v27 & 0xF;
    }
    v29 = (v27 >> 4) & 3;
    v30 = (v26 >> 4) & 3;
    if ( v29 == 1 || v30 == 1 )
    {
      v31 = 1;
    }
    else
    {
      if ( v29 != 2 && v30 != 2 )
      {
LABEL_52:
        v31 = 0;
        goto LABEL_53;
      }
      v31 = 2;
    }
LABEL_53:
    v9[32] = (16 * v31) | v9[32] & 0xCF;
    if ( (unsigned int)v28 - 7 <= 1 )
    {
LABEL_54:
      v9[33] |= 0x40u;
LABEL_55:
      v32 = (16 * v31) | a2[32] & 0xCF;
      a2[32] = v32;
      if ( (v32 & 0xFu) - 7 <= 1 || (v32 & 0x30) != 0 && (v32 & 0xF) != 9 )
        a2[33] |= 0x40u;
      if ( v9[32] >> 6 && (unsigned __int8)a2[32] >> 6 )
      {
        if ( v9[32] >> 6 == 1 || (v33 = 2, (unsigned __int8)a2[32] >> 6 == 1) )
          v33 = 1;
      }
      else
      {
        v33 = 0;
      }
      v9[32] = (v33 << 6) | v9[32] & 0x3F;
      a2[32] = (v33 << 6) | a2[32] & 0x3F;
      goto LABEL_13;
    }
LABEL_70:
    if ( (v9[32] & 0x30) == 0 || v28 == 9 )
      goto LABEL_55;
    goto LABEL_54;
  }
LABEL_13:
  LOBYTE(v13) = sub_15E4F60((__int64)a2);
  v14 = v13;
  if ( !(_BYTE)v13 )
  {
    v15 = sub_15E4F10((__int64)a2);
    if ( !v15 )
      goto LABEL_30;
    v16 = *(_QWORD **)(a1 + 160);
    v17 = (_QWORD *)(a1 + 152);
    if ( !v16 )
      goto LABEL_22;
    do
    {
      while ( 1 )
      {
        v18 = v16[2];
        v19 = v16[3];
        if ( v16[4] >= v15 )
          break;
        v16 = (_QWORD *)v16[3];
        if ( !v19 )
          goto LABEL_20;
      }
      v17 = v16;
      v16 = (_QWORD *)v16[2];
    }
    while ( v18 );
LABEL_20:
    if ( (_QWORD *)(a1 + 152) == v17 || v17[4] > v15 )
    {
LABEL_22:
      v39 = v17;
      v38 = a1 + 152;
      v20 = sub_22077B0(48);
      *(_QWORD *)(v20 + 32) = v15;
      *(_DWORD *)(v20 + 40) = 0;
      *(_BYTE *)(v20 + 44) = 0;
      v40 = v20;
      v21 = sub_167C8F0((_QWORD *)(a1 + 144), v39, (unsigned __int64 *)(v20 + 32));
      if ( v22 )
      {
        v23 = v38 == v22 || v21 || v15 < *(_QWORD *)(v22 + 32);
        sub_220F040(v23, v40, v22, v38);
        v17 = (_QWORD *)v40;
        ++*(_QWORD *)(a1 + 184);
      }
      else
      {
        v36 = v40;
        v41 = v21;
        j_j___libc_free_0(v36, 48);
        v17 = v41;
      }
    }
    if ( *((_BYTE *)v17 + 44) )
    {
LABEL_30:
      v42 = 1;
      if ( !v9 || (*(_BYTE *)(a1 + 72) & 1) != 0 )
        goto LABEL_62;
      v24 = sub_167C240(a1, (bool *)&v42, (__int64)v9, (__int64)a2);
      if ( (_BYTE)v24 )
        return v24;
      if ( v42 )
      {
LABEL_62:
        v43 = a2;
        sub_167D2C0(a1 + 16, &v43);
        return v14;
      }
    }
  }
  return 0;
}
