// Function: sub_F50FD0
// Address: 0xf50fd0
//
__int64 __fastcall sub_F50FD0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v4; // rax
  _QWORD *v5; // r12
  __int64 v6; // r12
  __int64 v7; // rcx
  _QWORD *v8; // r8
  __int64 v9; // r9
  __int64 v10; // r15
  __int64 v11; // rdx
  char v12; // cl
  __int64 v13; // rax
  int v14; // ecx
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // r13
  char *v17; // rdx
  _QWORD *v18; // rax
  unsigned int v19; // eax
  __int64 v20; // r15
  __int64 *v21; // rdx
  __int64 *v22; // rax
  char v23; // dl
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  __int64 v26; // r10
  __int64 *v27; // rax
  __int64 v28; // rax
  __int64 v29; // r13
  unsigned __int64 v30; // rdx
  __int64 *v31; // rax
  unsigned __int64 v32; // rax
  int v33; // edx
  _QWORD *v34; // rdi
  _QWORD *v35; // rax
  __int64 v36; // r13
  __int64 v37; // rsi
  _QWORD *v38; // rdi
  __int64 result; // rax
  char v40; // dl
  __int64 v41; // rax
  unsigned __int64 v42; // rdx
  __int64 *v43; // rax
  __int64 v44; // r13
  _QWORD *v45; // rax
  __int64 v46; // rax
  unsigned __int64 v47; // rax
  __int64 v48; // rsi
  __int64 v49; // rdx
  bool v50; // [rsp+17h] [rbp-289h]
  __int64 v51; // [rsp+20h] [rbp-280h]
  _QWORD *v52; // [rsp+20h] [rbp-280h]
  __int64 v53; // [rsp+20h] [rbp-280h]
  __int64 v54; // [rsp+20h] [rbp-280h]
  unsigned __int64 v55; // [rsp+28h] [rbp-278h]
  __int64 v56; // [rsp+28h] [rbp-278h]
  _QWORD *v57; // [rsp+28h] [rbp-278h]
  __int64 v58; // [rsp+30h] [rbp-270h] BYREF
  void *s; // [rsp+38h] [rbp-268h]
  _BYTE v60[12]; // [rsp+40h] [rbp-260h]
  unsigned __int8 v61; // [rsp+4Ch] [rbp-254h]
  char v62; // [rsp+50h] [rbp-250h] BYREF
  _BYTE *v63; // [rsp+60h] [rbp-240h] BYREF
  __int64 v64; // [rsp+68h] [rbp-238h]
  _BYTE v65[560]; // [rsp+70h] [rbp-230h] BYREF

  v2 = a2;
  v4 = *(_QWORD *)(a1 + 56);
  if ( !v4 )
    goto LABEL_90;
  if ( *(_BYTE *)(v4 - 24) == 84 )
  {
    while ( 1 )
    {
      v5 = (_QWORD *)(v4 - 24);
      a2 = **(_QWORD **)(v4 - 32);
      if ( a2 && (_QWORD *)a2 == v5 )
        a2 = sub_ACADE0(*(__int64 ***)(v4 - 16));
      sub_BD84D0((__int64)v5, a2);
      sub_B43D60(v5);
      v4 = *(_QWORD *)(a1 + 56);
      if ( !v4 )
        break;
      if ( *(_BYTE *)(v4 - 24) != 84 )
        goto LABEL_8;
    }
LABEL_90:
    BUG();
  }
LABEL_8:
  v6 = sub_AA54C0(a1);
  v50 = sub_AA5B70(v6);
  v63 = v65;
  v64 = 0x2000000000LL;
  if ( !v2 )
    goto LABEL_53;
  v58 = 0;
  s = &v62;
  *(_QWORD *)v60 = 2;
  *(_DWORD *)&v60[8] = 0;
  v61 = 1;
  v10 = *(_QWORD *)(v6 + 16);
  if ( !v10 )
  {
    v58 = 1;
    v16 = a1 & 0xFFFFFFFFFFFFFFFBLL;
    goto LABEL_34;
  }
  v11 = *(_QWORD *)(v6 + 16);
  while ( 1 )
  {
    v12 = **(_BYTE **)(v11 + 24);
    v13 = v11;
    v11 = *(_QWORD *)(v11 + 8);
    if ( (unsigned __int8)(v12 - 30) <= 0xAu )
      break;
    if ( !v11 )
      goto LABEL_17;
  }
  v14 = 0;
  while ( 1 )
  {
    v13 = *(_QWORD *)(v13 + 8);
    if ( !v13 )
      break;
    while ( (unsigned __int8)(**(_BYTE **)(v13 + 24) - 30) <= 0xAu )
    {
      v13 = *(_QWORD *)(v13 + 8);
      ++v14;
      if ( !v13 )
        goto LABEL_16;
    }
  }
LABEL_16:
  v15 = (unsigned int)(2 * v14 + 2) + 1LL;
  if ( v15 <= 0x20 )
  {
LABEL_17:
    v7 = 1;
    v16 = a1 & 0xFFFFFFFFFFFFFFFBLL;
    goto LABEL_18;
  }
  a2 = (__int64)v65;
  v16 = a1 & 0xFFFFFFFFFFFFFFFBLL;
  sub_C8D5F0((__int64)&v63, v65, v15, 0x10u, (__int64)v8, v9);
  v10 = *(_QWORD *)(v6 + 16);
  v7 = v61;
  if ( !v10 )
  {
    ++v58;
    goto LABEL_34;
  }
LABEL_18:
  while ( 1 )
  {
    v17 = *(char **)(v10 + 24);
    if ( (unsigned __int8)(*v17 - 30) <= 0xAu )
      break;
    v10 = *(_QWORD *)(v10 + 8);
    if ( !v10 )
    {
      ++v58;
LABEL_34:
      *(_QWORD *)&v60[4] = 0;
      goto LABEL_35;
    }
  }
  v8 = &v63;
  while ( 1 )
  {
    v9 = *((_QWORD *)v17 + 5);
    if ( v6 == v9 )
      goto LABEL_26;
    if ( !(_BYTE)v7 )
      goto LABEL_72;
    v18 = s;
    a2 = *(unsigned int *)&v60[4];
    v17 = (char *)s + 8 * *(unsigned int *)&v60[4];
    if ( s == v17 )
    {
LABEL_71:
      if ( *(_DWORD *)&v60[4] < *(_DWORD *)v60 )
      {
        a2 = (unsigned int)++*(_DWORD *)&v60[4];
        *(_QWORD *)v17 = v9;
        ++v58;
      }
      else
      {
LABEL_72:
        a2 = v9;
        v52 = v8;
        v56 = v9;
        sub_C8CC70((__int64)&v58, v9, (__int64)v17, v7, (__int64)v8, v9);
        v7 = v61;
        v9 = v56;
        v8 = v52;
        if ( !v40 )
          goto LABEL_26;
      }
      v41 = (unsigned int)v64;
      v42 = (unsigned int)v64 + 1LL;
      if ( v42 > HIDWORD(v64) )
      {
        a2 = (__int64)v65;
        v54 = v9;
        v57 = v8;
        sub_C8D5F0((__int64)v8, v65, v42, 0x10u, (__int64)v8, v9);
        v41 = (unsigned int)v64;
        v9 = v54;
        v8 = v57;
      }
      v43 = (__int64 *)&v63[16 * v41];
      *v43 = v9;
      v43[1] = v16;
      v7 = v61;
      LODWORD(v64) = v64 + 1;
      goto LABEL_26;
    }
    while ( v9 != *v18 )
    {
      if ( v17 == (char *)++v18 )
        goto LABEL_71;
    }
LABEL_26:
    v10 = *(_QWORD *)(v10 + 8);
    if ( !v10 )
      break;
    while ( 1 )
    {
      v17 = *(char **)(v10 + 24);
      if ( (unsigned __int8)(*v17 - 30) <= 0xAu )
        break;
      v10 = *(_QWORD *)(v10 + 8);
      if ( !v10 )
        goto LABEL_29;
    }
  }
LABEL_29:
  ++v58;
  if ( (_BYTE)v7 )
    goto LABEL_34;
  v19 = 4 * (*(_DWORD *)&v60[4] - *(_DWORD *)&v60[8]);
  if ( v19 < 0x20 )
    v19 = 32;
  if ( *(_DWORD *)v60 <= v19 )
  {
    a2 = 0xFFFFFFFFLL;
    memset(s, -1, 8LL * *(unsigned int *)v60);
    goto LABEL_34;
  }
  sub_C8C990((__int64)&v58, a2);
LABEL_35:
  v20 = *(_QWORD *)(v6 + 16);
  if ( v20 )
  {
    while ( 1 )
    {
      v21 = *(__int64 **)(v20 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v21 - 30) <= 0xAu )
        break;
      v20 = *(_QWORD *)(v20 + 8);
      if ( !v20 )
        goto LABEL_49;
    }
    v9 = v21[5];
    v55 = v6 & 0xFFFFFFFFFFFFFFFBLL;
    if ( !v61 )
      goto LABEL_45;
LABEL_38:
    v22 = (__int64 *)s;
    v7 = *(unsigned int *)&v60[4];
    v21 = (__int64 *)((char *)s + 8 * *(unsigned int *)&v60[4]);
    if ( s != v21 )
    {
      do
      {
        if ( v9 == *v22 )
          goto LABEL_42;
        ++v22;
      }
      while ( v21 != v22 );
    }
    if ( *(_DWORD *)&v60[4] < *(_DWORD *)v60 )
    {
      ++*(_DWORD *)&v60[4];
      *v21 = v9;
      ++v58;
      goto LABEL_46;
    }
LABEL_45:
    a2 = v9;
    v51 = v9;
    sub_C8CC70((__int64)&v58, v9, (__int64)v21, v7, (__int64)v8, v9);
    v9 = v51;
    if ( v23 )
    {
LABEL_46:
      v24 = (unsigned int)v64;
      v7 = HIDWORD(v64);
      v25 = (unsigned int)v64 + 1LL;
      v26 = v55 | 4;
      if ( v25 > HIDWORD(v64) )
      {
        a2 = (__int64)v65;
        v53 = v9;
        sub_C8D5F0((__int64)&v63, v65, v25, 0x10u, (__int64)v8, v9);
        v24 = (unsigned int)v64;
        v26 = v55 | 4;
        v9 = v53;
      }
      v27 = (__int64 *)&v63[16 * v24];
      *v27 = v9;
      v27[1] = v26;
      LODWORD(v64) = v64 + 1;
      v20 = *(_QWORD *)(v20 + 8);
      if ( v20 )
        goto LABEL_43;
    }
    else
    {
LABEL_42:
      while ( 1 )
      {
        v20 = *(_QWORD *)(v20 + 8);
        if ( !v20 )
          break;
LABEL_43:
        v21 = *(__int64 **)(v20 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v21 - 30) <= 0xAu )
        {
          v9 = v21[5];
          if ( v61 )
            goto LABEL_38;
          goto LABEL_45;
        }
      }
    }
  }
LABEL_49:
  v28 = (unsigned int)v64;
  v29 = v16 | 4;
  v30 = (unsigned int)v64 + 1LL;
  if ( v30 > HIDWORD(v64) )
  {
    a2 = (__int64)v65;
    sub_C8D5F0((__int64)&v63, v65, v30, 0x10u, (__int64)v8, v9);
    v28 = (unsigned int)v64;
  }
  v31 = (__int64 *)&v63[16 * v28];
  *v31 = v6;
  v31[1] = v29;
  LODWORD(v64) = v64 + 1;
  if ( !v61 )
    _libc_free(s, a2);
LABEL_53:
  if ( (*(_WORD *)(a1 + 2) & 0x7FFF) != 0 )
  {
    v44 = sub_ACC4F0(a1);
    v45 = (_QWORD *)sub_BD5C60(v44);
    v46 = sub_BCB2D0(v45);
    v47 = sub_ACD640(v46, 1, 0);
    v48 = sub_AD4C70(v47, *(__int64 ***)(v44 + 8), 0);
    sub_BD84D0(v44, v48);
    sub_ACFDF0((__int64 *)v44, v48, v49);
  }
  sub_BD84D0(v6, a1);
  v32 = *(_QWORD *)(v6 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v6 + 48 == v32 )
  {
    v34 = 0;
  }
  else
  {
    if ( !v32 )
      BUG();
    v33 = *(unsigned __int8 *)(v32 - 24);
    v34 = 0;
    v35 = (_QWORD *)(v32 - 24);
    if ( (unsigned int)(v33 - 30) < 0xB )
      v34 = v35;
  }
  sub_B43D60(v34);
  sub_AA80F0(a1, *(unsigned __int64 **)(a1 + 56), 1, v6, *(__int64 **)(v6 + 56), 1, (__int64 *)(v6 + 48), 0);
  v36 = sub_AA48A0(v6);
  sub_B43C20((__int64)&v58, v6);
  v37 = unk_3F148B8;
  v38 = sub_BD2C40(72, unk_3F148B8);
  if ( v38 )
  {
    v37 = v36;
    sub_B4C8A0((__int64)v38, v36, v58, (unsigned __int16)s);
  }
  if ( v50 )
  {
    v37 = v6;
    sub_AA4AF0(a1, v6);
    if ( !v2 )
      goto LABEL_77;
    sub_FFDB80(v2, v63, (unsigned int)v64);
    v37 = v6;
    result = sub_FFBF00(v2, v6);
    if ( *(_QWORD *)(v2 + 544) )
    {
      v37 = *(_QWORD *)(a1 + 72);
      result = sub_FFBD40(v2, v37);
    }
  }
  else
  {
    if ( v2 )
    {
      sub_FFDB80(v2, v63, (unsigned int)v64);
      v37 = v6;
      result = sub_FFBF00(v2, v6);
      goto LABEL_64;
    }
LABEL_77:
    result = sub_AA5450((_QWORD *)v6);
  }
LABEL_64:
  if ( v63 != v65 )
    return _libc_free(v63, v37);
  return result;
}
