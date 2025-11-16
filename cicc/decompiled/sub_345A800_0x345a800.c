// Function: sub_345A800
// Address: 0x345a800
//
unsigned __int8 *__fastcall sub_345A800(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, __m128i a5, __int64 a6, int a7)
{
  __int64 v7; // r8
  char v8; // r13
  __int64 v11; // rsi
  unsigned __int16 *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 *v15; // rdx
  __int64 v16; // r14
  __int64 v17; // r15
  unsigned __int8 *v18; // r13
  __int64 v20; // rdx
  __int128 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // r8
  int v24; // r9d
  unsigned int v25; // edx
  unsigned __int64 v26; // r15
  __int64 v27; // r9
  __int128 v28; // rax
  __int64 v29; // r9
  unsigned __int8 *v30; // rax
  __int64 v31; // rdx
  __int128 v32; // rax
  __int64 v33; // rcx
  __int64 v34; // r8
  int v35; // r9d
  unsigned int v36; // edx
  unsigned __int64 v37; // r15
  __int64 v38; // r9
  __int128 v39; // rax
  __int64 v40; // r9
  unsigned int v41; // edx
  __int64 v42; // rsi
  _BYTE *v43; // rax
  unsigned __int8 v44; // al
  __int64 v45; // rsi
  unsigned __int8 *v46; // rax
  unsigned __int16 v47; // bx
  unsigned __int8 *v48; // r14
  unsigned int v49; // edx
  unsigned __int64 v50; // r15
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // r8
  __int64 v54; // rdx
  __int64 v55; // rsi
  __int128 v56; // rax
  __int64 v57; // r9
  __int128 v58; // rax
  __int64 v59; // r9
  __int128 v60; // rax
  __int64 v61; // r9
  __int64 v62; // rdx
  __int128 v63; // rax
  __int64 v64; // rcx
  __int64 v65; // r8
  int v66; // r9d
  unsigned int v67; // edx
  unsigned __int64 v68; // r15
  __int64 v69; // r9
  __int128 v70; // rax
  __int64 v71; // r9
  __int128 v72; // [rsp-20h] [rbp-E0h]
  __int128 v73; // [rsp-20h] [rbp-E0h]
  __int128 v74; // [rsp-20h] [rbp-E0h]
  __int128 v75; // [rsp-20h] [rbp-E0h]
  __int128 v76; // [rsp-20h] [rbp-E0h]
  __int128 v77; // [rsp-20h] [rbp-E0h]
  __int128 v78; // [rsp-20h] [rbp-E0h]
  __int128 v79; // [rsp-20h] [rbp-E0h]
  __int128 v80; // [rsp+0h] [rbp-C0h]
  __int128 v81; // [rsp+0h] [rbp-C0h]
  __int128 v82; // [rsp+0h] [rbp-C0h]
  __int128 v83; // [rsp+0h] [rbp-C0h]
  unsigned __int8 *v84; // [rsp+20h] [rbp-A0h]
  unsigned __int8 *v85; // [rsp+30h] [rbp-90h]
  unsigned __int8 *v86; // [rsp+40h] [rbp-80h]
  __int64 v87; // [rsp+50h] [rbp-70h] BYREF
  int v88; // [rsp+58h] [rbp-68h]
  unsigned int v89; // [rsp+60h] [rbp-60h] BYREF
  __int64 v90; // [rsp+68h] [rbp-58h]
  unsigned __int16 v91; // [rsp+70h] [rbp-50h] BYREF
  __int64 v92; // [rsp+78h] [rbp-48h]

  v7 = a1;
  v8 = a4;
  v11 = *(_QWORD *)(a2 + 80);
  v87 = v11;
  if ( v11 )
  {
    sub_B96E90((__int64)&v87, v11, 1);
    v7 = a1;
  }
  v12 = *(unsigned __int16 **)(a2 + 48);
  v88 = *(_DWORD *)(a2 + 72);
  v13 = *v12;
  v14 = *((_QWORD *)v12 + 1);
  LOWORD(v89) = v13;
  v90 = v14;
  v15 = *(__int64 **)(a2 + 40);
  v16 = *v15;
  v17 = v15[1];
  if ( v8 )
  {
    if ( (_WORD)v13 == 1 )
    {
      if ( *(_BYTE *)(v7 + 6971) )
        goto LABEL_24;
      v62 = 1;
    }
    else
    {
      if ( !(_WORD)v13 )
      {
LABEL_6:
        if ( sub_30070B0((__int64)&v89) )
          goto LABEL_7;
        goto LABEL_32;
      }
      a4 = (unsigned __int16)v13;
      v62 = (unsigned __int16)v13;
      if ( !*(_QWORD *)(v7 + 8LL * (unsigned __int16)v13 + 112) )
        goto LABEL_24;
      a4 = 500LL * (unsigned __int16)v13;
      if ( *(_BYTE *)(v7 + a4 + 6471) )
        goto LABEL_24;
    }
    if ( !*(_BYTE *)(v7 + 500 * v62 + 6594) )
    {
      *(_QWORD *)&v63 = sub_3400BD0((__int64)a3, 0, (__int64)&v87, v89, v90, 0, a5, 0);
      v83 = v63;
      v84 = sub_33FB960((__int64)a3, v16, v17, a5, v64, v65, v66);
      v68 = v67 | v17 & 0xFFFFFFFF00000000LL;
      *((_QWORD *)&v78 + 1) = v68;
      *(_QWORD *)&v78 = v84;
      *(_QWORD *)&v70 = sub_3406EB0(a3, 0x39u, (__int64)&v87, v89, v90, v69, v83, v78);
      *((_QWORD *)&v79 + 1) = v68;
      *(_QWORD *)&v79 = v84;
      v30 = sub_3406EB0(a3, 0xB4u, (__int64)&v87, v89, v90, v71, v79, v70);
      goto LABEL_39;
    }
LABEL_24:
    if ( (unsigned __int16)(v13 - 17) <= 0xD3u )
    {
      v41 = (unsigned __int16)v13;
      v42 = v13 + 14;
      if ( !*(_QWORD *)(v7 + 8 * v13 + 112) )
        goto LABEL_7;
      v43 = (_BYTE *)(v7 + 500LL * (unsigned __int16)v13);
      if ( (v43[6605] & 0xFB) != 0 )
        goto LABEL_7;
      if ( v8 )
      {
        if ( !*(_QWORD *)(v7 + 8 * v42) || (v43[6471] & 0xFB) != 0 )
          goto LABEL_7;
      }
      else if ( !*(_QWORD *)(v7 + 8 * v42) || (v43[6470] & 0xFB) != 0 )
      {
        goto LABEL_7;
      }
      if ( !*(_QWORD *)(v7 + 8LL * (int)v41 + 112)
        || (a4 = 500LL * v41, v44 = *(_BYTE *)(v7 + a4 + 6602), v44 > 1u) && v44 != 4 )
      {
LABEL_7:
        v18 = 0;
        goto LABEL_8;
      }
    }
LABEL_32:
    v45 = v16;
    v46 = sub_33FB960((__int64)a3, v16, v17, a5, a4, v7, a7);
    v47 = v89;
    v48 = v46;
    v50 = v49 | v17 & 0xFFFFFFFF00000000LL;
    if ( (_WORD)v89 )
    {
      if ( (unsigned __int16)(v89 - 17) <= 0xD3u )
      {
        v54 = 0;
        v47 = word_4456580[(unsigned __int16)v89 - 1];
        goto LABEL_35;
      }
    }
    else if ( sub_30070B0((__int64)&v89) )
    {
      v47 = sub_3009970((__int64)&v89, v45, v51, v52, v53);
      goto LABEL_35;
    }
    v54 = v90;
LABEL_35:
    v91 = v47;
    v92 = v54;
    if ( v47 )
    {
      if ( v47 == 1 || (unsigned __int16)(v47 - 504) <= 7u )
        BUG();
      v55 = *(_QWORD *)&byte_444C4A0[16 * v47 - 16];
    }
    else
    {
      v55 = sub_3007260((__int64)&v91);
    }
    *(_QWORD *)&v56 = sub_3400E40((__int64)a3, v55 - 1, v89, v90, (__int64)&v87, a5);
    *((_QWORD *)&v76 + 1) = v50;
    *(_QWORD *)&v76 = v48;
    *(_QWORD *)&v58 = sub_3406EB0(a3, 0xBFu, (__int64)&v87, v89, v90, v57, v76, v56);
    *((_QWORD *)&v77 + 1) = v50;
    *(_QWORD *)&v77 = v48;
    v82 = v58;
    *(_QWORD *)&v60 = sub_3406EB0(a3, 0xBCu, (__int64)&v87, v89, v90, v59, v77, v58);
    if ( v8 )
      v30 = sub_3406EB0(a3, 0x39u, (__int64)&v87, v89, v90, v61, v82, v60);
    else
      v30 = sub_3406EB0(a3, 0x39u, (__int64)&v87, v89, v90, v61, v60, v82);
    goto LABEL_39;
  }
  if ( (_WORD)v13 == 1 )
  {
    if ( *(_BYTE *)(v7 + 6971) )
      goto LABEL_15;
    v20 = 1;
  }
  else
  {
    if ( !(_WORD)v13 )
      goto LABEL_6;
    a4 = (unsigned __int16)v13;
    v20 = (unsigned __int16)v13;
    if ( !*(_QWORD *)(v7 + 8LL * (unsigned __int16)v13 + 112) )
      goto LABEL_24;
    a4 = 500LL * (unsigned __int16)v13;
    if ( *(_BYTE *)(v7 + a4 + 6471) )
    {
LABEL_21:
      if ( (_WORD)v13 != 1 )
      {
        a4 = (unsigned __int16)v13;
        v31 = (unsigned __int16)v13;
        if ( !*(_QWORD *)(v7 + 8LL * (unsigned __int16)v13 + 112) )
          goto LABEL_24;
        a4 = 500LL * (unsigned __int16)v13;
        if ( *(_BYTE *)(v7 + a4 + 6471) )
          goto LABEL_24;
LABEL_17:
        if ( !*(_BYTE *)(v7 + 500 * v31 + 6596) )
        {
          *(_QWORD *)&v32 = sub_3400BD0((__int64)a3, 0, (__int64)&v87, v89, v90, 0, a5, 0);
          v81 = v32;
          v85 = sub_33FB960((__int64)a3, v16, v17, a5, v33, v34, v35);
          v37 = v36 | v17 & 0xFFFFFFFF00000000LL;
          *((_QWORD *)&v74 + 1) = v37;
          *(_QWORD *)&v74 = v85;
          *(_QWORD *)&v39 = sub_3406EB0(a3, 0x39u, (__int64)&v87, v89, v90, v38, v81, v74);
          *((_QWORD *)&v75 + 1) = v37;
          *(_QWORD *)&v75 = v85;
          v30 = sub_3406EB0(a3, 0xB6u, (__int64)&v87, v89, v90, v40, v75, v39);
          goto LABEL_39;
        }
        goto LABEL_24;
      }
LABEL_15:
      if ( *(_BYTE *)(v7 + 6971) )
        goto LABEL_24;
      v31 = 1;
      goto LABEL_17;
    }
  }
  if ( *(_BYTE *)(v7 + 500 * v20 + 6595) )
    goto LABEL_21;
  *(_QWORD *)&v21 = sub_3400BD0((__int64)a3, 0, (__int64)&v87, v89, v90, 0, a5, 0);
  v80 = v21;
  v86 = sub_33FB960((__int64)a3, v16, v17, a5, v22, v23, v24);
  v26 = v25 | v17 & 0xFFFFFFFF00000000LL;
  *((_QWORD *)&v72 + 1) = v26;
  *(_QWORD *)&v72 = v86;
  *(_QWORD *)&v28 = sub_3406EB0(a3, 0x39u, (__int64)&v87, v89, v90, v27, v80, v72);
  *((_QWORD *)&v73 + 1) = v26;
  *(_QWORD *)&v73 = v86;
  v30 = sub_3406EB0(a3, 0xB5u, (__int64)&v87, v89, v90, v29, v73, v28);
LABEL_39:
  v18 = v30;
LABEL_8:
  if ( v87 )
    sub_B91220((__int64)&v87, v87);
  return v18;
}
