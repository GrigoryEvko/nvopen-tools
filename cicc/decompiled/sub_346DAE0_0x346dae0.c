// Function: sub_346DAE0
// Address: 0x346dae0
//
__m128i *__fastcall sub_346DAE0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int16 *v5; // rax
  __int16 v6; // dx
  __int64 v7; // rax
  unsigned __int64 v8; // r15
  __int64 v9; // rax
  unsigned int v10; // edx
  __int64 *v11; // rbx
  __int64 v12; // rsi
  unsigned int v13; // ebx
  __int64 v14; // rdx
  __int64 v15; // r8
  unsigned __int16 v16; // ax
  char v17; // cl
  int v18; // esi
  __int64 v19; // rdx
  __int64 v20; // r9
  char v21; // di
  __int64 *v22; // r14
  unsigned __int16 v23; // ax
  __int64 v24; // r8
  unsigned int v25; // edx
  __int64 v26; // r9
  __int64 v27; // rsi
  char v28; // al
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rdx
  _QWORD *v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rsi
  _QWORD *v35; // rdx
  __int64 v36; // rax
  __int16 v37; // cx
  __int64 v38; // rax
  __m128i v39; // xmm0
  unsigned __int16 *v40; // rax
  __int64 v41; // rsi
  __int64 v42; // rdx
  __int64 v43; // r8
  __int64 v44; // r9
  unsigned __int8 v45; // al
  unsigned int v46; // edx
  unsigned int v47; // ebx
  __int64 v48; // r15
  __int64 v49; // rdx
  unsigned __int64 v50; // r15
  __int64 v51; // rax
  __int64 v52; // rdx
  __int128 v53; // rax
  __int64 v54; // r9
  __int128 v55; // rax
  __m128i v56; // xmm1
  __int64 v57; // rsi
  __int64 v58; // r8
  __int64 v59; // r9
  unsigned __int8 v60; // al
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // r8
  unsigned __int8 *v64; // r13
  unsigned int v65; // edx
  __int64 v66; // rbx
  __m128i *v67; // rax
  __m128i *v68; // r12
  unsigned __int64 v70; // rbx
  unsigned __int16 v71; // ax
  __int64 v72; // rdx
  __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rax
  __int128 v76; // rax
  __int64 v77; // r9
  __int64 v78; // rax
  unsigned int v79; // edx
  __int64 v80; // rsi
  __int64 v81; // rdx
  unsigned __int64 v82; // rsi
  __int64 v83; // rax
  __int128 v84; // rax
  __int64 v85; // r9
  unsigned int v86; // edx
  __m128i *v88; // [rsp+20h] [rbp-1F0h]
  unsigned int v89; // [rsp+2Ch] [rbp-1E4h]
  __int64 v90; // [rsp+30h] [rbp-1E0h]
  __int128 v91; // [rsp+30h] [rbp-1E0h]
  unsigned __int64 v92; // [rsp+40h] [rbp-1D0h]
  __m128i *v93; // [rsp+40h] [rbp-1D0h]
  __int64 v94; // [rsp+48h] [rbp-1C8h]
  unsigned __int64 v95; // [rsp+50h] [rbp-1C0h]
  __int128 v96; // [rsp+50h] [rbp-1C0h]
  __int128 v97; // [rsp+50h] [rbp-1C0h]
  __int64 *v98; // [rsp+60h] [rbp-1B0h]
  __int64 v99; // [rsp+68h] [rbp-1A8h]
  unsigned int v100; // [rsp+70h] [rbp-1A0h]
  __int128 v101; // [rsp+70h] [rbp-1A0h]
  __int128 v102; // [rsp+70h] [rbp-1A0h]
  unsigned __int64 v103; // [rsp+A0h] [rbp-170h]
  unsigned int v104; // [rsp+B0h] [rbp-160h] BYREF
  unsigned __int64 v105; // [rsp+B8h] [rbp-158h]
  __int64 v106; // [rsp+C0h] [rbp-150h] BYREF
  int v107; // [rsp+C8h] [rbp-148h]
  unsigned __int16 v108; // [rsp+D0h] [rbp-140h] BYREF
  __int64 v109; // [rsp+D8h] [rbp-138h]
  unsigned int v110; // [rsp+E0h] [rbp-130h] BYREF
  __int64 v111; // [rsp+E8h] [rbp-128h]
  unsigned __int64 v112; // [rsp+F0h] [rbp-120h]
  __int64 v113; // [rsp+F8h] [rbp-118h]
  __int64 v114; // [rsp+100h] [rbp-110h]
  __int64 v115; // [rsp+108h] [rbp-108h]
  __int64 v116; // [rsp+110h] [rbp-100h]
  __int64 v117; // [rsp+118h] [rbp-F8h]
  __int64 v118; // [rsp+120h] [rbp-F0h]
  __int64 v119; // [rsp+128h] [rbp-E8h]
  __int64 v120; // [rsp+130h] [rbp-E0h]
  __int64 v121; // [rsp+138h] [rbp-D8h]
  __int64 v122; // [rsp+150h] [rbp-C0h]
  __int64 v123; // [rsp+158h] [rbp-B8h]
  __m128i v124; // [rsp+160h] [rbp-B0h] BYREF
  __int64 v125; // [rsp+170h] [rbp-A0h]
  __int128 v126; // [rsp+180h] [rbp-90h] BYREF
  __int64 v127; // [rsp+190h] [rbp-80h]
  __int128 v128; // [rsp+1A0h] [rbp-70h] BYREF
  __int64 v129; // [rsp+1B0h] [rbp-60h]
  unsigned __int64 v130; // [rsp+1C0h] [rbp-50h] BYREF
  __int64 v131; // [rsp+1C8h] [rbp-48h]
  __int64 v132; // [rsp+1D0h] [rbp-40h]
  __int64 v133; // [rsp+1D8h] [rbp-38h]

  v5 = *(__int16 **)(a2 + 48);
  v6 = *v5;
  v105 = *((_QWORD *)v5 + 1);
  v7 = *(_QWORD *)(a2 + 40);
  LOWORD(v104) = v6;
  v8 = *(unsigned int *)(v7 + 8);
  v95 = *(_QWORD *)v7;
  v92 = *(_QWORD *)(v7 + 40);
  v89 = *(_DWORD *)(v7 + 48);
  v9 = *(_QWORD *)(*(_QWORD *)(v7 + 80) + 96LL);
  v10 = *(_DWORD *)(v9 + 32);
  v11 = *(__int64 **)(v9 + 24);
  if ( v10 > 0x40 )
  {
    v90 = *v11;
  }
  else
  {
    v90 = 0;
    if ( v10 )
      v90 = (__int64)((_QWORD)v11 << (64 - (unsigned __int8)v10)) >> (64 - (unsigned __int8)v10);
  }
  v12 = *(_QWORD *)(a2 + 80);
  v106 = v12;
  if ( v12 )
    sub_B96E90((__int64)&v106, v12, 1);
  v107 = *(_DWORD *)(a2 + 72);
  v13 = sub_33CD850((__int64)a3, v104, v105, 0);
  if ( (_WORD)v104 )
  {
    v21 = (unsigned __int16)(v104 - 176) <= 0x34u;
    v20 = 0;
    v29 = (unsigned __int16)v104 - 1;
    v17 = v21;
    v18 = 2 * word_4456340[v29];
    v16 = word_4456580[v29];
  }
  else
  {
    v103 = sub_3007240((__int64)&v104);
    v16 = sub_3009970((__int64)&v104, (unsigned int)(2 * v103), v14, HIDWORD(v103), v15);
    v17 = BYTE4(v103);
    v18 = 2 * v103;
    v20 = v19;
    v21 = BYTE4(v103);
  }
  LODWORD(v130) = v18;
  v22 = (__int64 *)a3[8];
  BYTE4(v130) = v17;
  v99 = v20;
  v100 = v16;
  if ( v21 )
  {
    v23 = sub_2D43AD0(v16, v18);
    v25 = v100;
    v26 = v99;
  }
  else
  {
    v23 = sub_2D43050(v16, v18);
    v26 = v99;
    v25 = v100;
  }
  if ( v23 )
  {
    v108 = v23;
    v109 = 0;
  }
  else
  {
    v23 = sub_3009450(v22, v25, v26, v130, v24, v26);
    v108 = v23;
    v109 = v30;
    if ( !v23 )
    {
      v116 = sub_3007260((__int64)&v108);
      v27 = v116;
      v117 = v31;
      v28 = v31;
      goto LABEL_18;
    }
  }
  if ( v23 == 1 || (unsigned __int16)(v23 - 504) <= 7u )
    goto LABEL_69;
  v27 = *(_QWORD *)&byte_444C4A0[16 * v23 - 16];
  v28 = byte_444C4A0[16 * v23 - 8];
LABEL_18:
  LOBYTE(v113) = v28;
  v112 = (unsigned __int64)(v27 + 7) >> 3;
  v32 = sub_33EDE90((__int64)a3, v112, v113, v13);
  v34 = v33;
  v35 = v32;
  *((_QWORD *)&v101 + 1) = v34;
  *(_QWORD *)&v101 = v32;
  v36 = v32[6] + 16LL * (unsigned int)v34;
  LODWORD(v35) = *((_DWORD *)v35 + 24);
  v37 = *(_WORD *)v36;
  v38 = *(_QWORD *)(v36 + 8);
  v98 = (__int64 *)a3[5];
  LOWORD(v110) = v37;
  v111 = v38;
  sub_2EAC300((__int64)&v124, (__int64)v98, (unsigned int)v35, 0);
  v130 = 0;
  v131 = 0;
  v39 = _mm_loadu_si128(&v124);
  v132 = 0;
  v133 = 0;
  v40 = (unsigned __int16 *)(*(_QWORD *)(v95 + 48) + 16LL * (unsigned int)v8);
  v129 = v125;
  v41 = *v40;
  v42 = *((_QWORD *)v40 + 1);
  v128 = (__int128)v39;
  v45 = sub_33CC4A0((__int64)a3, v41, v42, v95, v43, v44);
  v88 = sub_33F4560(
          a3,
          (unsigned __int64)(a3 + 36),
          0,
          (__int64)&v106,
          v95,
          v8,
          v101,
          *((unsigned __int64 *)&v101 + 1),
          v128,
          v129,
          v45,
          0,
          (__int64)&v130);
  v47 = v46;
  if ( (_WORD)v104 )
  {
    if ( (_WORD)v104 == 1 || (unsigned __int16)(v104 - 504) <= 7u )
      goto LABEL_69;
    v48 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v104 - 16];
  }
  else
  {
    v118 = sub_3007260((__int64)&v104);
    v48 = v118;
    v119 = v49;
  }
  v50 = (unsigned __int64)(v48 + 7) >> 3;
  if ( (_WORD)v110 )
  {
    if ( (_WORD)v110 == 1 || (unsigned __int16)(v110 - 504) <= 7u )
      goto LABEL_69;
    v51 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v110 - 16];
  }
  else
  {
    v51 = sub_3007260((__int64)&v110);
    v114 = v51;
    v115 = v52;
  }
  LODWORD(v131) = v51;
  if ( (unsigned int)v51 > 0x40 )
    sub_C43690((__int64)&v130, v50, 0);
  else
    v130 = v50;
  *(_QWORD *)&v53 = sub_3401900((__int64)a3, (__int64)&v106, v110, v111, (__int64)&v130, 1, v39);
  if ( (unsigned int)v131 > 0x40 && v130 )
  {
    v96 = v53;
    j_j___libc_free_0_0(v130);
    v53 = v96;
  }
  *(_QWORD *)&v55 = sub_3406EB0(a3, 0x38u, (__int64)&v106, v110, v111, v54, v101, v53);
  v130 = 0;
  v97 = v55;
  v131 = 0;
  v56 = _mm_loadu_si128(&v124);
  v132 = 0;
  v133 = 0;
  *(_QWORD *)&v55 = *(_QWORD *)(v92 + 48) + 16LL * v89;
  v129 = v125;
  v57 = *(unsigned __int16 *)v55;
  *((_QWORD *)&v55 + 1) = *(_QWORD *)(v55 + 8);
  v128 = (__int128)v56;
  v60 = sub_33CC4A0((__int64)a3, v57, *((__int64 *)&v55 + 1), v92, v58, v59);
  v93 = sub_33F4560(
          a3,
          (unsigned __int64)v88,
          v47,
          (__int64)&v106,
          v92,
          v89,
          v97,
          *((unsigned __int64 *)&v97 + 1),
          v128,
          v129,
          v60,
          0,
          (__int64)&v130);
  v94 = v61;
  if ( v90 >= 0 )
  {
    v64 = sub_3466750(a1, a3, v101, *((__int64 *)&v101 + 1), v104, v105, v39, *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
    v130 = 0;
    v131 = 0;
    v66 = v65;
    v132 = 0;
    v133 = 0;
    sub_2EAC3A0((__int64)&v126, v98);
    v67 = sub_33F1F00(
            a3,
            v104,
            v105,
            (__int64)&v106,
            (__int64)v93,
            v94,
            (__int64)v64,
            v66 | *((_QWORD *)&v101 + 1) & 0xFFFFFFFF00000000LL,
            v126,
            v127,
            0,
            0,
            (__int64)&v130,
            0);
    goto LABEL_29;
  }
  v70 = -v90;
  if ( (_WORD)v104 )
  {
    v71 = word_4456580[(unsigned __int16)v104 - 1];
    v72 = 0;
  }
  else
  {
    v71 = sub_3009970((__int64)&v104, (__int64)v88, v61, v62, v63);
  }
  LOWORD(v130) = v71;
  v131 = v72;
  if ( v71 )
  {
    if ( v71 == 1 || (unsigned __int16)(v71 - 504) <= 7u )
      goto LABEL_69;
    v74 = 16LL * (v71 - 1) + 71615648;
    v73 = *(_QWORD *)&byte_444C4A0[16 * v71 - 16];
    LOBYTE(v74) = *(_BYTE *)(v74 + 8);
  }
  else
  {
    v73 = sub_3007260((__int64)&v130);
    v120 = v73;
    v121 = v74;
  }
  LOBYTE(v131) = v74;
  v130 = v70 * ((unsigned __int64)(v73 + 7) >> 3);
  v75 = sub_CA1930(&v130);
  *(_QWORD *)&v76 = sub_3400BD0((__int64)a3, v75, (__int64)&v106, v110, v111, 0, v39, 0);
  v102 = v76;
  if ( !(_WORD)v104 )
  {
    if ( v70 <= (unsigned int)sub_3007240((__int64)&v104) )
      goto LABEL_40;
    v122 = sub_3007260((__int64)&v104);
    v80 = v122;
    v123 = v81;
LABEL_50:
    v82 = (unsigned __int64)(v80 + 7) >> 3;
    if ( !(_WORD)v110 )
    {
      LODWORD(v83) = sub_3007260((__int64)&v110);
LABEL_52:
      LODWORD(v131) = v83;
      if ( (unsigned int)v83 > 0x40 )
        sub_C43690((__int64)&v130, v82, 0);
      else
        v130 = v82;
      *(_QWORD *)&v84 = sub_3401900((__int64)a3, (__int64)&v106, v110, v111, (__int64)&v130, 1, v39);
      if ( (unsigned int)v131 > 0x40 && v130 )
      {
        v91 = v84;
        j_j___libc_free_0_0(v130);
        v84 = v91;
      }
      *(_QWORD *)&v102 = sub_3406EB0(a3, 0xB6u, (__int64)&v106, v110, v111, v85, v102, v84);
      *((_QWORD *)&v102 + 1) = v86 | *((_QWORD *)&v102 + 1) & 0xFFFFFFFF00000000LL;
      goto LABEL_40;
    }
    if ( (_WORD)v110 != 1 && (unsigned __int16)(v110 - 504) > 7u )
    {
      v83 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v110 - 16];
      goto LABEL_52;
    }
LABEL_69:
    BUG();
  }
  v78 = (unsigned __int16)v104 - 1;
  if ( v70 > word_4456340[v78] )
  {
    if ( (_WORD)v104 == 1 || (unsigned __int16)(v104 - 504) <= 7u )
      goto LABEL_69;
    v80 = *(_QWORD *)&byte_444C4A0[16 * v78];
    goto LABEL_50;
  }
LABEL_40:
  *(_QWORD *)&v97 = sub_3406EB0(a3, 0x39u, (__int64)&v106, v110, v111, v77, v97, v102);
  v130 = 0;
  v131 = 0;
  *((_QWORD *)&v97 + 1) = v79 | *((_QWORD *)&v97 + 1) & 0xFFFFFFFF00000000LL;
  v132 = 0;
  v133 = 0;
  sub_2EAC3A0((__int64)&v128, v98);
  v67 = sub_33F1F00(
          a3,
          v104,
          v105,
          (__int64)&v106,
          (__int64)v93,
          v94,
          v97,
          *((__int64 *)&v97 + 1),
          v128,
          v129,
          0,
          0,
          (__int64)&v130,
          0);
LABEL_29:
  v68 = v67;
  if ( v106 )
    sub_B91220((__int64)&v106, v106);
  return v68;
}
