// Function: sub_100F630
// Address: 0x100f630
//
unsigned __int8 *__fastcall sub_100F630(
        unsigned __int8 *a1,
        __int64 a2,
        __int64 a3,
        const __m128i *a4,
        unsigned __int8 a5,
        __int64 a6,
        int a7)
{
  int v7; // r14d
  __int64 *v8; // r13
  __int64 *v9; // rbx
  __int64 *v10; // rax
  unsigned __int8 *v11; // r15
  char v13; // al
  __int64 v14; // rax
  __int64 *v15; // r15
  __int64 v16; // rax
  unsigned __int8 **v17; // r10
  unsigned __int8 **v18; // r12
  __int64 v19; // rdx
  unsigned __int64 v20; // r9
  __int64 v21; // rsi
  __int64 v22; // rdx
  unsigned __int8 **v23; // rdi
  __int64 v24; // r9
  unsigned __int8 *v25; // r14
  __int64 v26; // r9
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  int v29; // r15d
  __int64 v30; // rcx
  char v31; // al
  int v32; // eax
  __int64 v33; // rdx
  __int64 v34; // r8
  unsigned __int8 *v35; // rax
  unsigned __int8 **v36; // r15
  unsigned __int8 **v37; // rbx
  unsigned __int8 *v38; // r13
  __int64 v39; // rax
  unsigned __int64 v40; // rdx
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r8
  unsigned __int8 *v45; // rax
  unsigned __int8 *v46; // r14
  unsigned __int8 *v47; // r14
  unsigned __int8 *v48; // rax
  unsigned __int8 **v49; // rax
  __int64 v50; // r9
  __int64 v51; // r8
  __int64 v52; // rcx
  __int64 v53; // rdx
  __int64 v54; // rdx
  __int64 *v55; // r15
  __int64 *v56; // r14
  __int64 *v57; // r13
  signed __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rax
  signed __int64 v61; // rdx
  __int64 v62; // rdx
  int v64; // [rsp+28h] [rbp-2F8h]
  unsigned int v65; // [rsp+30h] [rbp-2F0h]
  unsigned __int8 **v66; // [rsp+38h] [rbp-2E8h]
  bool v67; // [rsp+40h] [rbp-2E0h]
  unsigned __int8 *v68; // [rsp+40h] [rbp-2E0h]
  __int64 v69; // [rsp+48h] [rbp-2D8h]
  unsigned int v72; // [rsp+5Ch] [rbp-2C4h]
  __int64 *v73; // [rsp+60h] [rbp-2C0h]
  unsigned __int8 **v75; // [rsp+70h] [rbp-2B0h] BYREF
  __int64 v76; // [rsp+78h] [rbp-2A8h]
  _BYTE v77[64]; // [rsp+80h] [rbp-2A0h] BYREF
  _BYTE *v78; // [rsp+C0h] [rbp-260h] BYREF
  __int64 v79; // [rsp+C8h] [rbp-258h]
  _BYTE v80[64]; // [rsp+D0h] [rbp-250h] BYREF
  _BYTE *v81; // [rsp+110h] [rbp-210h] BYREF
  __int64 v82; // [rsp+118h] [rbp-208h]
  _BYTE v83[64]; // [rsp+120h] [rbp-200h] BYREF
  _BYTE *v84; // [rsp+160h] [rbp-1C0h] BYREF
  __int64 v85; // [rsp+168h] [rbp-1B8h]
  _BYTE v86[64]; // [rsp+170h] [rbp-1B0h] BYREF
  _BYTE *v87; // [rsp+1B0h] [rbp-170h] BYREF
  __int64 v88; // [rsp+1B8h] [rbp-168h]
  _BYTE v89[64]; // [rsp+1C0h] [rbp-160h] BYREF
  char *v90; // [rsp+200h] [rbp-120h] BYREF
  __int64 v91; // [rsp+208h] [rbp-118h]
  _BYTE v92[64]; // [rsp+210h] [rbp-110h] BYREF
  _BYTE *v93; // [rsp+250h] [rbp-D0h] BYREF
  __int64 v94; // [rsp+258h] [rbp-C8h]
  _BYTE v95[64]; // [rsp+260h] [rbp-C0h] BYREF
  char *v96; // [rsp+2A0h] [rbp-80h] BYREF
  __int64 v97; // [rsp+2A8h] [rbp-78h]
  _BYTE v98[112]; // [rsp+2B0h] [rbp-70h] BYREF

  v7 = a3;
  v8 = (__int64 *)a2;
  v9 = (__int64 *)(a2 + 16 * a3);
  v73 = (__int64 *)a2;
  v69 = 16 * a3;
  if ( (__int64 *)a2 != v9 )
  {
    v10 = (__int64 *)a2;
    do
    {
      a3 = *v10;
      if ( *(_BYTE *)*v10 <= 0x15u )
        return 0;
      if ( a1 == (unsigned __int8 *)a3 )
        return (unsigned __int8 *)v10[1];
      v10 += 2;
    }
    while ( v9 != v10 );
  }
  if ( !a7 )
    return 0;
  v13 = *a1;
  LOBYTE(a3) = *a1 == 84;
  v67 = v13 == 84 || (unsigned __int8)v13 <= 0x1Cu;
  if ( v67 )
    return 0;
  if ( v13 == 85 )
  {
    v14 = *((_QWORD *)a1 - 4);
    if ( v14 )
    {
      if ( !*(_BYTE *)v14 )
      {
        a2 = *((_QWORD *)a1 + 10);
        if ( *(_QWORD *)(v14 + 24) == a2 && *(_DWORD *)(v14 + 36) == 206 )
          return 0;
      }
    }
  }
  else if ( v13 == 96 )
  {
    return 0;
  }
  v15 = v8;
  if ( v8 != v9 )
  {
    while ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(*v15 + 8) + 8LL) - 17 > 1
         || (unsigned __int8)sub_98C610((char *)a1, a2, a3) )
    {
      v15 += 2;
      if ( v9 == v15 )
        goto LABEL_18;
    }
    return 0;
  }
LABEL_18:
  v75 = (unsigned __int8 **)v77;
  v76 = 0x800000000LL;
  v16 = 4LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
  if ( (a1[7] & 0x40) != 0 )
  {
    v17 = (unsigned __int8 **)*((_QWORD *)a1 - 1);
    v66 = &v17[v16];
  }
  else
  {
    v66 = (unsigned __int8 **)a1;
    v17 = (unsigned __int8 **)&a1[-(v16 * 8)];
  }
  if ( v17 == v66 )
    return 0;
  v18 = v17;
  v64 = v7;
  v65 = a7 - 1;
  do
  {
    v25 = *v18;
    v21 = (__int64)v8;
    v45 = (unsigned __int8 *)sub_100F630((unsigned int)*v18, (_DWORD)v8, v64, (_DWORD)a4, a5, a6, v65);
    if ( v45 )
    {
      v19 = (unsigned int)v76;
      v20 = (unsigned int)v76 + 1LL;
      if ( v20 > HIDWORD(v76) )
      {
        v68 = v45;
        sub_C8D5F0((__int64)&v75, v77, (unsigned int)v76 + 1LL, 8u, v44, v20);
        v19 = (unsigned int)v76;
        v45 = v68;
      }
      v67 = v45 != v25;
      v75[v19] = v45;
      v21 = (unsigned int)v76;
      v22 = (unsigned int)(v76 + 1);
      LODWORD(v76) = v76 + 1;
    }
    else
    {
      v27 = (unsigned int)v76;
      v28 = (unsigned int)v76 + 1LL;
      if ( v28 > HIDWORD(v76) )
      {
        v21 = (__int64)v77;
        sub_C8D5F0((__int64)&v75, v77, v28, 8u, v44, v26);
        v27 = (unsigned int)v76;
      }
      v75[v27] = v25;
      v22 = (unsigned int)(v76 + 1);
      LODWORD(v76) = v76 + 1;
    }
    v23 = v75;
    v24 = (unsigned int)v22;
    if ( (unsigned int)*v75[(unsigned int)v22 - 1] - 12 <= 1 && !a4[4].m128i_i8[1] )
      goto LABEL_65;
    v18 += 4;
  }
  while ( v66 != v18 );
  if ( !v67 )
    goto LABEL_65;
  if ( a5 )
  {
    v21 = (__int64)v75;
    v48 = sub_100EA40(a1, v75, (unsigned int)v22, a4, v65, (unsigned int)v22);
    v23 = v75;
    v11 = v48;
    if ( v48 == a1 )
      v11 = 0;
    goto LABEL_63;
  }
  v29 = *a1;
  v30 = (unsigned int)(v29 - 42);
  v31 = *a1;
  if ( (unsigned int)v30 > 0x11 )
  {
LABEL_50:
    if ( v31 == 63 && (_DWORD)v22 == 2 )
    {
      if ( (unsigned __int8)sub_FFFE90((__int64)v23[1]) )
        goto LABEL_70;
      v22 = (unsigned int)v76;
      v23 = v75;
    }
    v36 = &v23[v22];
    v37 = v23;
    v96 = v98;
    v97 = 0x800000000LL;
    if ( v36 == v23 )
    {
LABEL_56:
      v21 = a6 == 0;
      if ( !sub_98CD70(a1, v21)
        || sub_988010((__int64)a1)
        && (unsigned int)sub_987FE0((__int64)a1) == 1
        && (unsigned __int8)sub_AD7CA0(*(_BYTE **)v96, v21, v41, v42, v43) )
      {
        v21 = (__int64)v96;
        v59 = sub_97D230(a1, (__int64 *)v96, (unsigned int)v97, a4->m128i_i64[0], (__int64 *)a4->m128i_i64[1], 0);
        v11 = (unsigned __int8 *)v59;
        if ( a6
          && v59
          && ((unsigned __int8)sub_B44920(a1)
           || (unsigned __int8)sub_B44AB0(a1)
           || (unsigned __int8)sub_B44930((__int64)a1)) )
        {
          v21 = (__int64)a1;
          sub_9C95B0(a6, (__int64)a1);
        }
        goto LABEL_60;
      }
    }
    else
    {
      while ( 1 )
      {
        v38 = *v37;
        if ( **v37 > 0x15u )
          break;
        v39 = (unsigned int)v97;
        v40 = (unsigned int)v97 + 1LL;
        if ( v40 > HIDWORD(v97) )
        {
          v21 = (__int64)v98;
          sub_C8D5F0((__int64)&v96, v98, v40, 8u, v44, v24);
          v39 = (unsigned int)v97;
        }
        ++v37;
        *(_QWORD *)&v96[8 * v39] = v38;
        LODWORD(v97) = v97 + 1;
        if ( v36 == v37 )
          goto LABEL_56;
      }
    }
    v11 = 0;
LABEL_60:
    if ( v96 != v98 )
      _libc_free(v96, v21);
    v23 = v75;
    goto LABEL_63;
  }
  v21 = *((_QWORD *)a1 + 1);
  v72 = v29 - 29;
  v32 = *(unsigned __int8 *)(v21 + 8);
  v33 = (unsigned int)(v32 - 17);
  if ( (unsigned int)v33 <= 1 )
    LOBYTE(v32) = *(_BYTE *)(**(_QWORD **)(v21 + 16) + 8LL);
  if ( (unsigned __int8)v32 > 3u && (_BYTE)v32 != 5 && (v32 & 0xFD) != 4 )
  {
    v46 = *v75;
    if ( v46 == sub_AD93D0(v72, v21, 0, 0) )
    {
      v23 = v75;
      v11 = v75[1];
      goto LABEL_63;
    }
    v21 = *((_QWORD *)a1 + 1);
    v47 = v75[1];
    if ( v47 == sub_AD93D0(v72, v21, 1, 0) )
    {
LABEL_70:
      v23 = v75;
      v11 = *v75;
      goto LABEL_63;
    }
  }
  v34 = (unsigned int)(v29 - 57);
  if ( (unsigned int)v34 <= 1 )
  {
    v49 = v75;
    v21 = (__int64)v75[1];
    v23 = v75;
    if ( *v75 != (unsigned __int8 *)v21 )
      goto LABEL_47;
    if ( *a1 != 58 || (a1[1] & 2) == 0 )
    {
LABEL_79:
      v11 = *v49;
      v23 = v49;
      goto LABEL_63;
    }
    if ( a6 )
    {
      v21 = (__int64)a1;
      sub_9C95B0(a6, (__int64)a1);
      v49 = v75;
      goto LABEL_79;
    }
LABEL_65:
    v11 = 0;
    goto LABEL_63;
  }
  if ( v29 != 44 && v29 != 59 || *v75 != v75[1] )
    goto LABEL_47;
  v78 = v80;
  v79 = 0x800000000LL;
  if ( (_DWORD)v76 )
    sub_FFEBC0((__int64)&v78, (__int64)&v75, v33, v30, v34, v24);
  v50 = (unsigned int)v79;
  v81 = v83;
  v82 = 0x800000000LL;
  if ( (_DWORD)v79 )
    sub_FFEBC0((__int64)&v81, (__int64)&v78, v33, v30, v34, (unsigned int)v79);
  v51 = (unsigned int)v82;
  v84 = v86;
  v85 = 0x800000000LL;
  if ( (_DWORD)v82 )
    sub_FFEBC0((__int64)&v84, (__int64)&v81, v33, v30, (unsigned int)v82, v50);
  v87 = v89;
  v88 = 0x800000000LL;
  if ( (_DWORD)v85 )
    sub_FFEBC0((__int64)&v87, (__int64)&v84, v33, v30, v51, v50);
  v21 = (unsigned int)v88;
  v90 = v92;
  v91 = 0x800000000LL;
  if ( (_DWORD)v88 )
  {
    v21 = (__int64)&v87;
    sub_FFEBC0((__int64)&v90, (__int64)&v87, v33, v30, v51, (__int64)&v90);
  }
  v52 = (unsigned int)v91;
  v96 = v98;
  v97 = 0x800000000LL;
  if ( (_DWORD)v91 )
  {
    v21 = (__int64)&v90;
    sub_FFE870((__int64)&v96, &v90, v33, (unsigned int)v91, v51, (__int64)&v90);
  }
  v53 = (unsigned int)v97;
  v93 = v95;
  v94 = 0x800000000LL;
  if ( (_DWORD)v97 )
  {
    v21 = (__int64)&v96;
    sub_FFE870((__int64)&v93, &v96, (unsigned int)v97, v52, v51, (__int64)&v93);
  }
  if ( v96 != v98 )
    _libc_free(v96, v21);
  v96 = v98;
  v97 = 0x800000000LL;
  if ( (_DWORD)v94 )
  {
    v21 = (__int64)&v93;
    sub_FFEBC0((__int64)&v96, (__int64)&v93, v53, v52, v51, (__int64)&v93);
  }
  if ( v69 >> 6 > 0 )
  {
    v54 = *(_QWORD *)v96;
    v55 = v8;
    while ( v54 != v55[1] )
    {
      if ( v54 == v55[3] )
      {
        v55 += 2;
        goto LABEL_106;
      }
      if ( v54 == v55[5] )
      {
        v55 += 4;
        goto LABEL_106;
      }
      if ( v54 == v55[7] )
      {
        v55 += 6;
        goto LABEL_106;
      }
      v55 += 8;
      if ( &v8[8 * (v69 >> 6)] == v55 )
        goto LABEL_154;
    }
    goto LABEL_106;
  }
  v55 = v8;
LABEL_154:
  v61 = (char *)v9 - (char *)v55;
  if ( (char *)v9 - (char *)v55 == 32 )
  {
    v62 = *(_QWORD *)v96;
LABEL_169:
    if ( v62 == v55[1] )
      goto LABEL_106;
    v55 += 2;
    goto LABEL_165;
  }
  if ( v61 == 48 )
  {
    v62 = *(_QWORD *)v96;
    if ( *(_QWORD *)v96 == v55[1] )
      goto LABEL_106;
    v55 += 2;
    goto LABEL_169;
  }
  if ( v61 == 16 )
  {
    v62 = *(_QWORD *)v96;
LABEL_165:
    if ( v62 == v55[1] )
      goto LABEL_106;
  }
  v55 = v9;
LABEL_106:
  if ( v96 != v98 )
    _libc_free(v96, v21);
  if ( v93 != v95 )
    _libc_free(v93, v21);
  if ( v90 != v92 )
    _libc_free(v90, v21);
  if ( v87 != v89 )
    _libc_free(v87, v21);
  if ( v84 != v86 )
    _libc_free(v84, v21);
  if ( v81 != v83 )
    _libc_free(v81, v21);
  if ( v9 != v55 )
  {
    if ( v78 != v80 )
      _libc_free(v78, v21);
    v60 = sub_AD6530(*((_QWORD *)a1 + 1), v21);
    v23 = v75;
    v11 = (unsigned __int8 *)v60;
    goto LABEL_63;
  }
  if ( v78 != v80 )
    _libc_free(v78, v21);
LABEL_47:
  v21 = *((_QWORD *)a1 + 1);
  v35 = (unsigned __int8 *)sub_AD6840(v72, v21, 0);
  v23 = v75;
  v11 = v35;
  if ( *v75 != v35 && v75[1] != v35 )
    goto LABEL_49;
  if ( v69 >> 6 <= 0 )
  {
LABEL_136:
    v58 = (char *)v9 - (char *)v73;
    if ( (char *)v9 - (char *)v73 != 32 )
    {
      if ( v58 != 48 )
      {
        if ( v58 != 16 )
          goto LABEL_139;
        goto LABEL_162;
      }
      v21 = *v73;
      if ( (unsigned __int8)sub_98EF70((__int64)a1, *v73) )
        goto LABEL_133;
      v73 += 2;
    }
    v21 = *v73;
    if ( (unsigned __int8)sub_98EF70((__int64)a1, *v73) )
      goto LABEL_133;
    v73 += 2;
LABEL_162:
    v21 = *v73;
    if ( (unsigned __int8)sub_98EF70((__int64)a1, *v73) )
      goto LABEL_133;
LABEL_139:
    v31 = *a1;
    v22 = (unsigned int)v76;
    v23 = v75;
    goto LABEL_50;
  }
  v56 = v73;
  v57 = &v8[8 * (v69 >> 6)];
  while ( 1 )
  {
    v21 = *v56;
    if ( (unsigned __int8)sub_98EF70((__int64)a1, *v56) )
    {
      v73 = v56;
      goto LABEL_133;
    }
    v21 = v56[2];
    if ( (unsigned __int8)sub_98EF70((__int64)a1, v21) )
      break;
    v21 = v56[4];
    if ( (unsigned __int8)sub_98EF70((__int64)a1, v21) )
    {
      v73 = v56 + 4;
      goto LABEL_133;
    }
    v21 = v56[6];
    if ( (unsigned __int8)sub_98EF70((__int64)a1, v21) )
    {
      v73 = v56 + 6;
      goto LABEL_133;
    }
    v56 += 8;
    if ( v57 == v56 )
    {
      v73 = v56;
      goto LABEL_136;
    }
  }
  v73 = v56 + 2;
LABEL_133:
  v23 = v75;
  if ( v9 == v73 )
  {
LABEL_49:
    v31 = *a1;
    v22 = (unsigned int)v76;
    goto LABEL_50;
  }
LABEL_63:
  if ( v23 != (unsigned __int8 **)v77 )
    _libc_free(v23, v21);
  return v11;
}
