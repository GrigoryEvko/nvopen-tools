// Function: sub_1FA2D60
// Address: 0x1fa2d60
//
__int64 __fastcall sub_1FA2D60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v7; // rax
  __int64 v8; // rsi
  __m128i v9; // xmm0
  __m128i v10; // xmm1
  __m128i v11; // xmm2
  __int64 v12; // r14
  __int64 v13; // rcx
  __int64 v14; // r15
  __int64 v15; // rbx
  int v16; // eax
  __int64 v17; // r9
  const void ***v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rsi
  unsigned int v21; // edx
  __int64 v22; // rbx
  unsigned __int8 v23; // al
  const void **v24; // rbx
  __int64 v25; // r14
  __int64 *v26; // rax
  int v27; // eax
  unsigned __int8 *v29; // rax
  __int64 v30; // r14
  __int128 v31; // rax
  __int64 v32; // rax
  __int64 *v33; // rbx
  int v34; // edx
  __int64 v35; // r15
  __int128 v36; // rax
  __int64 *v37; // rax
  int v38; // edx
  __int64 v39; // r11
  __int64 v40; // rcx
  bool v41; // dl
  bool v42; // r9
  _DWORD *v43; // rsi
  bool v44; // dl
  int v45; // eax
  __int64 v46; // rax
  __int64 v47; // rax
  bool v48; // r9
  __int64 *v49; // r15
  const void ***v50; // rax
  __int128 v51; // rax
  __int64 v52; // r9
  __int64 *v53; // rax
  int v54; // edx
  __int64 *v55; // rax
  int v56; // edx
  bool v57; // al
  __int64 v58; // rax
  unsigned int v59; // r15d
  int v60; // eax
  bool v61; // al
  __int64 v62; // rax
  unsigned int v63; // r15d
  int v64; // eax
  unsigned int v65; // edx
  __int128 v66; // [rsp+0h] [rbp-F0h]
  __int64 v67; // [rsp+18h] [rbp-D8h]
  __int64 v68; // [rsp+20h] [rbp-D0h]
  bool v69; // [rsp+28h] [rbp-C8h]
  _QWORD *v70; // [rsp+30h] [rbp-C0h]
  bool v71; // [rsp+38h] [rbp-B8h]
  bool v72; // [rsp+38h] [rbp-B8h]
  __int64 v73; // [rsp+38h] [rbp-B8h]
  __int64 v74; // [rsp+38h] [rbp-B8h]
  bool v75; // [rsp+38h] [rbp-B8h]
  _QWORD *v76; // [rsp+38h] [rbp-B8h]
  bool v77; // [rsp+38h] [rbp-B8h]
  bool v78; // [rsp+43h] [rbp-ADh]
  unsigned int v79; // [rsp+44h] [rbp-ACh]
  __int64 v80; // [rsp+48h] [rbp-A8h]
  unsigned __int8 v81; // [rsp+48h] [rbp-A8h]
  unsigned int v82; // [rsp+48h] [rbp-A8h]
  const void **v83; // [rsp+50h] [rbp-A0h]
  __int128 *v84; // [rsp+50h] [rbp-A0h]
  __int128 v85; // [rsp+60h] [rbp-90h]
  int v86; // [rsp+60h] [rbp-90h]
  __int128 *v87; // [rsp+70h] [rbp-80h]
  __int64 v88; // [rsp+70h] [rbp-80h]
  __int64 v89; // [rsp+80h] [rbp-70h] BYREF
  int v90; // [rsp+88h] [rbp-68h]
  _QWORD *v91; // [rsp+90h] [rbp-60h] BYREF
  unsigned int v92; // [rsp+98h] [rbp-58h]
  __int64 v93; // [rsp+A0h] [rbp-50h] BYREF
  const void **v94; // [rsp+A8h] [rbp-48h]
  __int64 *v95; // [rsp+B0h] [rbp-40h]
  int v96; // [rsp+B8h] [rbp-38h]

  v7 = *(__int64 **)(a2 + 32);
  v8 = *(_QWORD *)(a2 + 72);
  v9 = _mm_loadu_si128((const __m128i *)v7);
  v10 = _mm_loadu_si128((const __m128i *)(v7 + 5));
  v11 = _mm_loadu_si128((const __m128i *)v7 + 5);
  v12 = *v7;
  v13 = v7[5];
  v79 = *((_DWORD *)v7 + 2);
  v14 = v7[10];
  v15 = *((unsigned int *)v7 + 22);
  v89 = v8;
  if ( v8 )
  {
    v80 = v13;
    sub_1623A60((__int64)&v89, v8, 2);
    v13 = v80;
  }
  v90 = *(_DWORD *)(a2 + 64);
  v16 = *(unsigned __int16 *)(v12 + 24);
  if ( v16 == 32 || v16 == 10 )
  {
    v27 = *(unsigned __int16 *)(v13 + 24);
    if ( v27 != 10 && v27 != 32 )
    {
      v25 = (__int64)sub_1D37470(
                       *(__int64 **)a1,
                       68,
                       (__int64)&v89,
                       *(const void ****)(a2 + 40),
                       *(_DWORD *)(a2 + 60),
                       a6,
                       *(_OWORD *)&v10,
                       *(_OWORD *)&v9,
                       *(_OWORD *)&v11);
      goto LABEL_21;
    }
  }
  if ( sub_1D185B0(v11.m128i_i64[0]) )
  {
    v18 = *(const void ****)(a2 + 40);
    if ( !*(_BYTE *)(a1 + 24)
      || ((v19 = *(unsigned __int8 *)v18, v20 = *(_QWORD *)(a1 + 8), v21 = 1, (_BYTE)v19 == 1)
       || (_BYTE)v19 && (v21 = (unsigned __int8)v19, *(_QWORD *)(v20 + 8 * v19 + 120)))
      && (*(_BYTE *)(v20 + 259LL * v21 + 2493) & 0xFB) == 0 )
    {
      v25 = (__int64)sub_1D37440(
                       *(__int64 **)a1,
                       71,
                       (__int64)&v89,
                       v18,
                       *(_DWORD *)(a2 + 60),
                       v17,
                       *(double *)v9.m128i_i64,
                       *(double *)v10.m128i_i64,
                       v11,
                       *(_OWORD *)&v9,
                       *(_OWORD *)&v10);
      goto LABEL_21;
    }
  }
  v22 = *(_QWORD *)(v14 + 40) + 16 * v15;
  v23 = *(_BYTE *)v22;
  v24 = *(const void ***)(v22 + 8);
  v81 = v23;
  if ( sub_1D185B0(v9.m128i_i64[0]) && sub_1D185B0(v10.m128i_i64[0]) )
  {
    v29 = (unsigned __int8 *)(*(_QWORD *)(v12 + 40) + 16LL * v79);
    v30 = v81;
    *((_QWORD *)&v66 + 1) = v24;
    *(_QWORD *)&v66 = v81;
    v82 = *v29;
    v83 = (const void **)*((_QWORD *)v29 + 1);
    *(_QWORD *)&v31 = sub_1D32610(
                        *(__int64 **)a1,
                        v11.m128i_i64[0],
                        v11.m128i_i64[1],
                        (__int64)&v89,
                        *v29,
                        v83,
                        *(double *)v9.m128i_i64,
                        *(double *)v10.m128i_i64,
                        *(double *)v11.m128i_i64,
                        v66);
    v85 = v31;
    sub_1F81BC0(a1, v31);
    v32 = sub_1D38BB0(*(_QWORD *)a1, 0, (__int64)&v89, v30, v24, 0, v9, *(double *)v10.m128i_i64, v11, 0);
    v33 = *(__int64 **)a1;
    LODWORD(v30) = v34;
    v35 = v32;
    *(_QWORD *)&v36 = sub_1D38BB0(*(_QWORD *)a1, 1, (__int64)&v89, v82, v83, 0, v9, *(double *)v10.m128i_i64, v11, 0);
    v37 = sub_1D332F0(
            v33,
            118,
            (__int64)&v89,
            v82,
            v83,
            0,
            *(double *)v9.m128i_i64,
            *(double *)v10.m128i_i64,
            v11,
            v85,
            *((unsigned __int64 *)&v85 + 1),
            v36);
    LODWORD(v94) = v38;
    v95 = (__int64 *)v35;
    v96 = v30;
    v93 = (__int64)v37;
    v25 = sub_1F994A0(a1, a2, &v93, 2, 1);
    goto LABEL_21;
  }
  if ( !sub_1D18970(v9.m128i_i64[0]) )
    goto LABEL_14;
  if ( !sub_1D185B0(v10.m128i_i64[0]) )
    goto LABEL_14;
  if ( *(_WORD *)(v14 + 24) != 120 )
    goto LABEL_14;
  v39 = *(_QWORD *)(v14 + 32);
  v40 = *(_QWORD *)(v39 + 40);
  v41 = *(_WORD *)(v40 + 24) == 32 || *(_WORD *)(v40 + 24) == 10;
  v42 = v41;
  if ( !v41 )
    goto LABEL_14;
  v43 = *(_DWORD **)(a1 + 8);
  v94 = v24;
  LOBYTE(v93) = v81;
  if ( v81 )
  {
    if ( (unsigned __int8)(v81 - 14) > 0x5Fu )
    {
      v44 = (unsigned __int8)(v81 - 86) <= 0x17u || (unsigned __int8)(v81 - 8) <= 5u;
      goto LABEL_32;
    }
LABEL_47:
    v45 = v43[17];
    goto LABEL_34;
  }
  v78 = v41;
  v67 = v40;
  v68 = v39;
  v72 = sub_1F58CD0((__int64)&v93);
  v57 = sub_1F58D20((__int64)&v93);
  v44 = v72;
  v39 = v68;
  v40 = v67;
  v42 = v78;
  if ( v57 )
    goto LABEL_47;
LABEL_32:
  if ( v44 )
    v45 = v43[16];
  else
    v45 = v43[15];
LABEL_34:
  if ( v45 == 1 )
  {
    v62 = *(_QWORD *)(v40 + 88);
    v63 = *(_DWORD *)(v62 + 32);
    if ( v63 <= 0x40 )
    {
      v61 = *(_QWORD *)(v62 + 24) == 1;
    }
    else
    {
      v74 = v39;
      v64 = sub_16A57B0(v62 + 24);
      v39 = v74;
      v61 = v63 - 1 == v64;
    }
LABEL_50:
    if ( v61 )
      goto LABEL_45;
    goto LABEL_14;
  }
  if ( v45 == 2 )
  {
    v58 = *(_QWORD *)(v40 + 88);
    v59 = *(_DWORD *)(v58 + 32);
    if ( v59 <= 0x40 )
    {
      v61 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v59) == *(_QWORD *)(v58 + 24);
    }
    else
    {
      v73 = v39;
      v60 = sub_16A58F0(v58 + 24);
      v39 = v73;
      v61 = v59 == v60;
    }
    goto LABEL_50;
  }
  v46 = *(_QWORD *)(v40 + 88);
  LODWORD(v94) = *(_DWORD *)(v46 + 32);
  if ( (unsigned int)v94 <= 0x40 )
  {
    v93 = *(_QWORD *)(v46 + 24);
LABEL_38:
    LODWORD(v94) = 0;
    v47 = v93 & 1;
    v93 = v47;
LABEL_39:
    v48 = v47 == 1;
    goto LABEL_40;
  }
  v75 = v42;
  sub_16A4FD0((__int64)&v93, (const void **)(v46 + 24));
  if ( (unsigned int)v94 <= 0x40 )
    goto LABEL_38;
  v69 = v75;
  *(_QWORD *)v93 &= 1uLL;
  v76 = (_QWORD *)v93;
  memset((void *)(v93 + 8), 0, 8 * (unsigned int)(((unsigned __int64)(unsigned int)v94 + 63) >> 6) - 8);
  v65 = (unsigned int)v94;
  LODWORD(v94) = 0;
  v92 = v65;
  v47 = (__int64)v76;
  v91 = v76;
  v70 = v76;
  if ( v65 <= 0x40 )
    goto LABEL_39;
  if ( v65 - (unsigned int)sub_16A57B0((__int64)&v91) <= 0x40 && (v48 = v69, *v76 == 1) || (v48 = 0, v76) )
  {
    v77 = v48;
    j_j___libc_free_0_0(v70);
    v48 = v77;
  }
LABEL_40:
  if ( (unsigned int)v94 > 0x40 && v93 )
  {
    v71 = v48;
    j_j___libc_free_0_0(v93);
    v48 = v71;
  }
  if ( v48 )
  {
    v39 = *(_QWORD *)(v14 + 32);
LABEL_45:
    v49 = *(__int64 **)a1;
    v84 = (__int128 *)v39;
    v87 = *(__int128 **)(v12 + 32);
    v50 = (const void ***)(*(_QWORD *)(v12 + 40) + 16LL * v79);
    *(_QWORD *)&v51 = sub_1D38BB0(
                        *(_QWORD *)a1,
                        0,
                        (__int64)&v89,
                        *(unsigned __int8 *)v50,
                        v50[1],
                        0,
                        v9,
                        *(double *)v10.m128i_i64,
                        v11,
                        0);
    v53 = sub_1D37470(v49, 69, (__int64)&v89, *(const void ****)(a2 + 40), *(_DWORD *)(a2 + 60), v52, v51, *v87, *v84);
    v86 = v54;
    v88 = (__int64)v53;
    v55 = sub_1F6DC60(
            (__int64)v53,
            1u,
            (__int64)&v89,
            v81,
            v24,
            *(__int64 **)a1,
            v9,
            *(double *)v10.m128i_i64,
            v11,
            *(_DWORD **)(a1 + 8));
    v96 = v56;
    v93 = v88;
    LODWORD(v94) = v86;
    v95 = v55;
    v25 = sub_1F994A0(a1, a2, &v93, 2, 1);
    goto LABEL_21;
  }
LABEL_14:
  v25 = (__int64)sub_1F83290(
                   a1,
                   v9.m128i_i64[0],
                   v9.m128i_i64[1],
                   v10.m128i_i64[0],
                   v10.m128i_u64[1],
                   a2,
                   v9,
                   *(double *)v10.m128i_i64,
                   v11,
                   *(_OWORD *)&v11);
  if ( !v25 )
  {
    v26 = sub_1F83290(
            a1,
            v10.m128i_i64[0],
            v10.m128i_i64[1],
            v9.m128i_i64[0],
            v9.m128i_u64[1],
            a2,
            v9,
            *(double *)v10.m128i_i64,
            v11,
            *(_OWORD *)&v11);
    if ( v26 )
      v25 = (__int64)v26;
  }
LABEL_21:
  if ( v89 )
    sub_161E7C0((__int64)&v89, v89);
  return v25;
}
