// Function: sub_3813E70
// Address: 0x3813e70
//
unsigned __int8 *__fastcall sub_3813E70(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        __m128i a7,
        __int64 a8,
        __int64 a9,
        unsigned int a10)
{
  unsigned __int16 *v12; // rdx
  int v13; // eax
  __int64 v14; // r12
  int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // rsi
  unsigned int v18; // esi
  unsigned __int16 v19; // r10
  unsigned __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r9
  bool v23; // al
  unsigned __int64 v24; // rax
  int v25; // esi
  bool v26; // cl
  unsigned __int16 v27; // ax
  __int64 v28; // r8
  unsigned int v29; // edx
  __int64 v30; // r9
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  unsigned __int8 *v34; // rax
  unsigned int v35; // edx
  __int64 v36; // r15
  unsigned __int8 *v37; // rax
  unsigned int v38; // edx
  __int64 v39; // rdx
  unsigned __int8 *v40; // rsi
  unsigned int v41; // r13d
  unsigned int v42; // eax
  __int128 v43; // rdi
  unsigned __int8 *v44; // rax
  int v45; // edx
  int v46; // esi
  unsigned __int8 *v47; // rdx
  unsigned __int8 *v48; // r14
  unsigned __int8 *v50; // rax
  unsigned int v51; // edx
  unsigned __int16 v52; // ax
  __int64 v53; // rdx
  __int128 v54; // [rsp-20h] [rbp-100h]
  unsigned __int16 v55; // [rsp+8h] [rbp-D8h]
  __int64 v56; // [rsp+8h] [rbp-D8h]
  __int64 v57; // [rsp+10h] [rbp-D0h]
  unsigned int v58; // [rsp+10h] [rbp-D0h]
  unsigned __int16 v59; // [rsp+10h] [rbp-D0h]
  __int64 *v60; // [rsp+18h] [rbp-C8h]
  __int64 v61; // [rsp+18h] [rbp-C8h]
  char v62; // [rsp+24h] [rbp-BCh]
  char v63; // [rsp+28h] [rbp-B8h]
  __int64 v64; // [rsp+28h] [rbp-B8h]
  bool v66; // [rsp+37h] [rbp-A9h]
  unsigned int v67; // [rsp+38h] [rbp-A8h]
  __int64 v70; // [rsp+70h] [rbp-70h]
  unsigned int v71; // [rsp+80h] [rbp-60h] BYREF
  __int64 v72; // [rsp+88h] [rbp-58h]
  __int64 v73; // [rsp+90h] [rbp-50h] BYREF
  __int64 v74; // [rsp+98h] [rbp-48h]

  v12 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3);
  v13 = *v12;
  v14 = *((_QWORD *)v12 + 1);
  LOWORD(v71) = v13;
  v72 = v14;
  if ( (_WORD)v13 )
  {
    if ( (unsigned __int16)(v13 - 17) > 0xD3u )
    {
      LOWORD(v73) = v13;
      v74 = v14;
LABEL_4:
      if ( (_WORD)v13 == 1 || (unsigned __int16)(v13 - 504) <= 7u )
        BUG();
      v67 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v13 - 16];
      v15 = *(_DWORD *)(a1 + 24);
      if ( v15 == 92 )
        goto LABEL_7;
      goto LABEL_11;
    }
    LOWORD(v13) = word_4456580[v13 - 1];
    v16 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v71) )
    {
      v74 = v14;
      LOWORD(v73) = 0;
      goto LABEL_10;
    }
    LOWORD(v13) = sub_3009970((__int64)&v71, a2, v31, v32, v33);
  }
  LOWORD(v73) = v13;
  v74 = v16;
  if ( (_WORD)v13 )
    goto LABEL_4;
LABEL_10:
  v67 = sub_3007260((__int64)&v73);
  v15 = *(_DWORD *)(a1 + 24);
  if ( v15 == 92 )
  {
LABEL_7:
    v62 = 1;
    v63 = 1;
LABEL_13:
    v66 = v15 == 95;
    goto LABEL_14;
  }
LABEL_11:
  if ( v15 != 94 )
  {
    v62 = 0;
    v63 = 0;
    goto LABEL_13;
  }
  v62 = 1;
  v63 = 1;
  v66 = 1;
LABEL_14:
  v17 = *(_QWORD *)(a1 + 80);
  v73 = v17;
  if ( v17 )
    sub_B96E90((__int64)&v73, v17, 1);
  LODWORD(v74) = *(_DWORD *)(a1 + 72);
  v18 = 2 * v67;
  if ( 2 * v67 == 2 )
  {
    v19 = 3;
    goto LABEL_33;
  }
  switch ( v18 )
  {
    case 4u:
      v19 = 4;
      goto LABEL_33;
    case 8u:
      v19 = 5;
      goto LABEL_33;
    case 0x10u:
      v19 = 6;
      goto LABEL_33;
    case 0x20u:
      v19 = 7;
      goto LABEL_33;
    case 0x40u:
      v19 = 8;
      goto LABEL_33;
    case 0x80u:
      v19 = 9;
LABEL_33:
      LODWORD(v20) = (unsigned __int16)v71;
      v22 = 0;
      if ( !(_WORD)v71 )
        goto LABEL_24;
      goto LABEL_34;
  }
  v19 = sub_3007020(*(_QWORD **)(a9 + 64), v18);
  LODWORD(v20) = (unsigned __int16)v71;
  v22 = v21;
  if ( !(_WORD)v71 )
  {
LABEL_24:
    v55 = v19;
    v57 = v22;
    v23 = sub_30070B0((__int64)&v71);
    v22 = v57;
    v19 = v55;
    if ( !v23 )
      goto LABEL_35;
    v24 = sub_3007240((__int64)&v71);
    v19 = v55;
    v22 = v57;
    v25 = v24;
    v20 = HIDWORD(v24);
    v26 = v20;
    goto LABEL_26;
  }
LABEL_34:
  if ( (unsigned __int16)(v20 - 17) > 0xD3u )
    goto LABEL_35;
  v26 = (unsigned __int16)(v20 - 176) <= 0x34u;
  v25 = word_4456340[(int)v20 - 1];
  LOBYTE(v20) = v26;
LABEL_26:
  LODWORD(v70) = v25;
  BYTE4(v70) = v20;
  v60 = *(__int64 **)(a9 + 64);
  v56 = v22;
  v58 = v19;
  if ( v26 )
  {
    v27 = sub_2D43AD0(v19, v25);
    v29 = v58;
    v30 = v56;
    v19 = v27;
    if ( v27 )
    {
LABEL_28:
      v22 = 0;
      goto LABEL_35;
    }
  }
  else
  {
    v52 = sub_2D43050(v19, v25);
    v30 = v56;
    v29 = v58;
    v19 = v52;
    if ( v52 )
      goto LABEL_28;
  }
  v19 = sub_3009450(v60, v29, v30, v70, v28, v30);
  v22 = v53;
LABEL_35:
  v59 = v19;
  v61 = v22;
  if ( v63 )
  {
    v34 = sub_33FB160(a9, a2, a3, (__int64)&v73, v19, v22, a7);
    v36 = v35;
    v64 = (__int64)v34;
    v37 = sub_33FB160(a9, a4, a5, (__int64)&v73, v59, v61, a7);
  }
  else
  {
    v50 = sub_33FB310(a9, a2, a3, (__int64)&v73, v19, v22, a7);
    v36 = v51;
    v64 = (__int64)v50;
    v37 = sub_33FB310(a9, a4, a5, (__int64)&v73, v59, v61, a7);
  }
  *((_QWORD *)&v54 + 1) = v38 | a5 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v54 = v37;
  v40 = sub_34696C0(a8, *(_DWORD *)(a1 + 24), (__int64)&v73, v64, v36 | a3 & 0xFFFFFFFF00000000LL, a6, a7, v54, a9);
  v41 = v39;
  if ( v66 )
  {
    v42 = v67;
    *(_QWORD *)&v43 = v40;
    *((_QWORD *)&v43 + 1) = v39;
    if ( a10 )
      v42 = a10;
    v44 = sub_38139C0(v43, (__int64)&v73, v42, v62, (_QWORD *)a9, a7);
    v46 = v45;
    v47 = v44;
    LODWORD(v44) = v46;
    v40 = v47;
    v41 = (unsigned int)v44;
  }
  v48 = sub_33FB310(a9, (__int64)v40, v41, (__int64)&v73, v71, v72, a7);
  if ( v73 )
    sub_B91220((__int64)&v73, v73);
  return v48;
}
