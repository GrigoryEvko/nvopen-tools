// Function: sub_26FFED0
// Address: 0x26ffed0
//
void __fastcall sub_26FFED0(__int64 a1, __int64 **a2)
{
  __int64 *v2; // rbx
  __int64 *v3; // rax
  __int64 *v4; // r15
  __int64 *i; // rax
  unsigned __int8 *v6; // r14
  __int64 *v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned int *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rbx
  __int64 v14; // rbx
  _QWORD *v15; // rax
  __int64 v16; // rcx
  __int64 **v17; // rax
  __int64 v18; // rbx
  const void *v19; // rsi
  __int64 v20; // rdx
  unsigned __int64 v21; // rax
  int v22; // edx
  __int64 *v23; // r13
  __int64 v24; // rcx
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 *v27; // r12
  __int64 *v28; // rax
  __int64 *v29; // rdx
  char v30; // di
  _DWORD *v31; // rax
  __int64 *k; // r12
  unsigned __int64 v33; // rdi
  __int64 v34; // rax
  _QWORD *v35; // r13
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rbx
  __int64 v39; // rax
  int v40; // ebx
  __int64 v41; // rax
  __int64 v42; // rdx
  unsigned __int8 *v43; // rbx
  __int64 v44; // rax
  unsigned __int8 *v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // r8
  __int64 v49; // rdx
  __int64 *v50; // rax
  __int64 v51; // rbx
  __int64 v52; // r13
  __int64 v53; // r12
  _QWORD *v54; // rax
  __int64 v55; // rbx
  unsigned int *v56; // r13
  unsigned int *v57; // r12
  __int64 v58; // rdx
  unsigned int v59; // esi
  __int64 v60; // rax
  __int64 ***v61; // rax
  __int64 v62; // rax
  char *v63; // rsi
  int j; // r12d
  char *v65; // rsi
  __int64 v66; // rax
  __int64 v67; // r12
  unsigned __int64 v68; // rax
  unsigned __int64 v69; // rax
  __int64 *v70; // rax
  unsigned __int8 *v71; // rax
  char *v72; // rax
  __int64 v73; // rdx
  __int64 v74; // r9
  unsigned __int64 v75; // rsi
  unsigned __int64 v76; // rcx
  __int64 *v77; // r13
  _QWORD *v78; // rcx
  unsigned __int8 *v79; // rax
  __int64 v80; // rsi
  __int64 *v81; // rax
  unsigned __int64 v82; // rdi
  __int64 v83; // [rsp+10h] [rbp-210h]
  int v84; // [rsp+18h] [rbp-208h]
  unsigned __int8 *v85; // [rsp+18h] [rbp-208h]
  unsigned int v86; // [rsp+2Ch] [rbp-1F4h]
  __int64 v87; // [rsp+38h] [rbp-1E8h]
  __int64 v88; // [rsp+38h] [rbp-1E8h]
  char *v89; // [rsp+38h] [rbp-1E8h]
  size_t n; // [rsp+58h] [rbp-1C8h]
  size_t na; // [rsp+58h] [rbp-1C8h]
  __int64 *v92; // [rsp+88h] [rbp-198h]
  _QWORD *v93; // [rsp+90h] [rbp-190h]
  __int64 *v95; // [rsp+A0h] [rbp-180h]
  __int64 v96; // [rsp+B8h] [rbp-168h] BYREF
  __int64 v97; // [rsp+C0h] [rbp-160h] BYREF
  __int64 v98; // [rsp+C8h] [rbp-158h] BYREF
  _QWORD v99[4]; // [rsp+D0h] [rbp-150h] BYREF
  __int16 v100; // [rsp+F0h] [rbp-130h]
  char *v101; // [rsp+100h] [rbp-120h] BYREF
  char *v102; // [rsp+108h] [rbp-118h]
  char *v103; // [rsp+110h] [rbp-110h]
  __int16 v104; // [rsp+120h] [rbp-100h]
  __int64 v105; // [rsp+130h] [rbp-F0h] BYREF
  __int64 v106; // [rsp+138h] [rbp-E8h] BYREF
  __int64 *v107; // [rsp+140h] [rbp-E0h]
  __int64 *v108; // [rsp+148h] [rbp-D8h]
  __int64 *v109; // [rsp+150h] [rbp-D0h]
  __int64 v110; // [rsp+158h] [rbp-C8h]
  unsigned int *v111; // [rsp+160h] [rbp-C0h] BYREF
  __int64 v112; // [rsp+168h] [rbp-B8h]
  _BYTE v113[32]; // [rsp+170h] [rbp-B0h] BYREF
  __int64 v114; // [rsp+190h] [rbp-90h]
  __int64 v115; // [rsp+198h] [rbp-88h]
  __int64 v116; // [rsp+1A0h] [rbp-80h]
  __int64 v117; // [rsp+1A8h] [rbp-78h]
  void **v118; // [rsp+1B0h] [rbp-70h]
  void **v119; // [rsp+1B8h] [rbp-68h]
  __int64 v120; // [rsp+1C0h] [rbp-60h]
  int v121; // [rsp+1C8h] [rbp-58h]
  __int16 v122; // [rsp+1CCh] [rbp-54h]
  char v123; // [rsp+1CEh] [rbp-52h]
  __int64 v124; // [rsp+1D0h] [rbp-50h]
  __int64 v125; // [rsp+1D8h] [rbp-48h]
  void *v126; // [rsp+1E0h] [rbp-40h] BYREF
  void *v127; // [rsp+1E8h] [rbp-38h] BYREF

  v2 = a2[1];
  v108 = &v106;
  v109 = &v106;
  v3 = *a2;
  LODWORD(v106) = 0;
  v107 = 0;
  v110 = 0;
  v95 = v2;
  if ( v3 == v2 )
  {
    v33 = 0;
    goto LABEL_47;
  }
  v4 = v3;
  for ( i = 0; ; i = v107 )
  {
    v6 = (unsigned __int8 *)v4[1];
    if ( !i )
      goto LABEL_109;
    v7 = &v106;
    do
    {
      while ( 1 )
      {
        v8 = i[2];
        v9 = i[3];
        if ( i[4] >= (unsigned __int64)v6 )
          break;
        i = (__int64 *)i[3];
        if ( !v9 )
          goto LABEL_8;
      }
      v7 = i;
      i = (__int64 *)i[2];
    }
    while ( v8 );
LABEL_8:
    if ( v7 == &v106 || v7[4] > (unsigned __int64)v6 )
    {
LABEL_109:
      v10 = sub_B491C0(v4[1]);
      v96 = sub_B2D7E0(v10, "target-features", 0xFu);
      if ( v96 )
      {
        v11 = (unsigned int *)sub_A72240(&v96);
        v112 = v12;
        v111 = v11;
        if ( sub_C931B0((__int64 *)&v111, "+retpoline", 0xAu, 0) != -1 )
        {
          v13 = *(_QWORD *)(a1 + 8);
          if ( *(_BYTE *)(v13 + 104) )
          {
            v71 = sub_BD3990(**(unsigned __int8 ***)(a1 + 16), (__int64)"+retpoline");
            v72 = (char *)sub_BD5D20((__int64)v71);
            sub_26F96D0(
              (__int64)v4,
              "branch-funnel",
              13,
              v72,
              v73,
              v74,
              *(__int64 (__fastcall **)(__int64, __int64))(v13 + 112),
              *(_QWORD *)(v13 + 120));
            v13 = *(_QWORD *)(a1 + 8);
          }
          v14 = *(_QWORD *)(v13 + 64);
          v15 = (_QWORD *)sub_22077B0(8u);
          v93 = v15;
          if ( v15 )
            *v15 = v14;
          v16 = *((_QWORD *)v6 + 10);
          v17 = *(__int64 ***)(v16 + 16);
          v18 = 8LL * *(unsigned int *)(v16 + 12);
          v19 = v17 + 1;
          if ( v17 + 1 == &v17[(unsigned __int64)v18 / 8] || (v20 = v18 - 8, v18 == 8) )
          {
            v21 = sub_BCF480(*v17, v93, 1, *(_DWORD *)(v16 + 8) >> 8 != 0);
          }
          else
          {
            if ( (unsigned __int64)v20 > 0x7FFFFFFFFFFFFFF0LL )
              goto LABEL_104;
            v34 = v20 >> 3;
            if ( !(v20 >> 3) )
              v34 = 1;
            v35 = (_QWORD *)sub_22077B0(8 * v34 + 8);
            *v35 = *v93;
            memcpy(v35 + 1, v19, v18 - 8);
            j_j___libc_free_0((unsigned __int64)v93);
            v93 = v35;
            v21 = sub_BCF480(
                    **(__int64 ***)(*((_QWORD *)v6 + 10) + 16LL),
                    v35,
                    v18 >> 3,
                    *(_DWORD *)(*((_QWORD *)v6 + 10) + 8LL) >> 8 != 0);
          }
          n = v21;
          v117 = sub_BD5C60((__int64)v6);
          v118 = &v126;
          v119 = &v127;
          v111 = (unsigned int *)v113;
          v126 = &unk_49DA100;
          v112 = 0x200000000LL;
          v122 = 512;
          LOWORD(v116) = 0;
          v127 = &unk_49DA0B0;
          v120 = 0;
          v121 = 0;
          v123 = 7;
          v124 = 0;
          v125 = 0;
          v114 = 0;
          v115 = 0;
          sub_D5F1F0((__int64)&v111, (__int64)v6);
          v92 = (__int64 *)sub_22077B0(8u);
          if ( v92 )
            *v92 = *v4;
          v22 = *v6;
          v23 = v92 + 1;
          if ( v22 == 40 )
          {
            v24 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)v6);
          }
          else
          {
            v24 = -32;
            if ( v22 != 85 )
            {
              if ( v22 != 34 )
                goto LABEL_105;
              v24 = -96;
            }
          }
          if ( (v6[7] & 0x80u) != 0 )
          {
            v87 = v24;
            v36 = sub_BD2BC0((__int64)v6);
            v24 = v87;
            v38 = v36 + v37;
            if ( (v6[7] & 0x80u) == 0 )
            {
              if ( (unsigned int)(v38 >> 4) )
LABEL_105:
                BUG();
            }
            else
            {
              v39 = sub_BD2BC0((__int64)v6);
              v24 = v87;
              if ( (unsigned int)((v38 - v39) >> 4) )
              {
                if ( (v6[7] & 0x80u) == 0 )
                  goto LABEL_105;
                v40 = *(_DWORD *)(sub_BD2BC0((__int64)v6) + 8);
                if ( (v6[7] & 0x80u) == 0 )
                  BUG();
                v41 = sub_BD2BC0((__int64)v6);
                v24 = v87 - 32LL * (unsigned int)(*(_DWORD *)(v41 + v42 - 4) - v40);
              }
            }
          }
          v43 = &v6[v24];
          v44 = 32LL * (*((_DWORD *)v6 + 1) & 0x7FFFFFF);
          v45 = &v6[-v44];
          if ( &v6[-v44] != &v6[v24] )
          {
            v46 = v24 + v44;
            if ( v46 >> 5 )
            {
              if ( v46 < 0 )
LABEL_104:
                sub_4262D8((__int64)"vector::_M_range_insert");
              v75 = 0x7FFFFFFFFFFFFFF8LL;
              v76 = (v46 >> 5) + 1;
              if ( v46 >> 5 != -1 )
              {
                if ( v76 > 0xFFFFFFFFFFFFFFFLL )
                  v76 = 0xFFFFFFFFFFFFFFFLL;
                v75 = 8 * v76;
              }
              v85 = v45;
              v77 = (__int64 *)sub_22077B0(v75);
              v78 = v77 + 1;
              *v77 = *v92;
              v79 = v85;
              do
              {
                v80 = *(_QWORD *)v79;
                v79 += 32;
                *v78++ = v80;
              }
              while ( v43 != v79 );
              j_j___libc_free_0((unsigned __int64)v92);
              v92 = v77;
              v49 = (__int64)(8 * ((unsigned __int64)(v43 - v85 - 32) >> 5) + 16) >> 3;
              v48 = v49;
              goto LABEL_63;
            }
            do
            {
              v47 = *(_QWORD *)v45;
              v45 += 32;
              *v23++ = v47;
            }
            while ( v43 != v45 );
          }
          v48 = 1;
          LODWORD(v49) = 1;
LABEL_63:
          v50 = *(__int64 **)(a1 + 16);
          if ( *v6 == 85 )
          {
            v104 = 257;
            v55 = sub_921880(&v111, n, *v50, (int)v92, v48, (__int64)&v101, 0);
          }
          else
          {
            v83 = v48;
            v100 = 257;
            v51 = *((_QWORD *)v6 - 8);
            v52 = *((_QWORD *)v6 - 12);
            v53 = *v50;
            v104 = 257;
            v88 = v51;
            v84 = v49 + 3;
            v54 = sub_BD2CC0(88, (unsigned int)(v49 + 3));
            v55 = (__int64)v54;
            if ( v54 )
            {
              v86 = v84 & 0x7FFFFFF | v86 & 0xE0000000;
              sub_B44260((__int64)v54, **(_QWORD **)(n + 16), 5, v86, 0, 0);
              *(_QWORD *)(v55 + 72) = 0;
              sub_B4A9C0(v55, n, v53, v52, v88, (__int64)&v101, v92, v83, 0, 0);
            }
            if ( (_BYTE)v122 )
            {
              v81 = (__int64 *)sub_BD5C60(v55);
              *(_QWORD *)(v55 + 72) = sub_A7A090((__int64 *)(v55 + 72), v81, -1, 72);
            }
            (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v119 + 2))(
              v119,
              v55,
              v99,
              v115,
              v116);
            v56 = v111;
            v57 = &v111[4 * (unsigned int)v112];
            if ( v111 != v57 )
            {
              do
              {
                v58 = *((_QWORD *)v56 + 1);
                v59 = *v56;
                v56 += 4;
                sub_B99FD0(v55, v59, v58);
              }
              while ( v57 != v56 );
            }
          }
          *(_WORD *)(v55 + 2) = *((_WORD *)v6 + 1) & 0xFFC | *(_WORD *)(v55 + 2) & 0xF003;
          v60 = *((_QWORD *)v6 + 9);
          v101 = 0;
          v97 = v60;
          v61 = *(__int64 ****)(a1 + 8);
          v102 = 0;
          v103 = 0;
          v98 = sub_A778C0(**v61, 21, 0);
          v62 = sub_A79C90(***(__int64 ****)(a1 + 8), &v98, 1);
          v63 = v102;
          v99[0] = v62;
          if ( v102 == v103 )
          {
            sub_10E63E0(&v101, v102, v99);
          }
          else
          {
            if ( v102 )
            {
              *(_QWORD *)v102 = v62;
              v63 = v102;
            }
            v102 = v63 + 8;
          }
          for ( j = 0; (unsigned int)sub_A74480((__int64)&v97) > j + 2; ++j )
          {
            v66 = sub_A744E0(&v97, j);
            v65 = v102;
            v99[0] = v66;
            if ( v102 == v103 )
            {
              sub_10E63E0(&v101, v102, v99);
            }
            else
            {
              if ( v102 )
              {
                *(_QWORD *)v102 = v66;
                v65 = v102;
              }
              v102 = v65 + 8;
            }
          }
          v67 = v102 - v101;
          v89 = v101;
          na = sub_A74610(&v97);
          v68 = sub_A74680(&v97);
          v69 = sub_A78180(***(_QWORD ****)(a1 + 8), v68, na, v89, v67 >> 3);
          v27 = &v106;
          *(_QWORD *)(v55 + 72) = v69;
          v70 = v107;
          if ( !v107 )
            goto LABEL_27;
          do
          {
            if ( v70[4] < (unsigned __int64)v6 )
            {
              v70 = (__int64 *)v70[3];
            }
            else
            {
              v27 = v70;
              v70 = (__int64 *)v70[2];
            }
          }
          while ( v70 );
          if ( v27 == &v106 || v27[4] > (unsigned __int64)v6 )
          {
LABEL_27:
            v25 = sub_22077B0(0x30u);
            v26 = (__int64)v27;
            *(_QWORD *)(v25 + 32) = v6;
            v27 = (__int64 *)v25;
            *(_QWORD *)(v25 + 40) = 0;
            v28 = sub_26FFDD0(&v105, v26, (unsigned __int64 *)(v25 + 32));
            if ( v29 )
            {
              v30 = v28 || &v106 == v29 || (unsigned __int64)v6 < v29[4];
              sub_220F040(v30, (__int64)v27, v29, &v106);
              ++v110;
            }
            else
            {
              v82 = (unsigned __int64)v27;
              v27 = v28;
              j_j___libc_free_0(v82);
            }
          }
          v27[5] = v55;
          v31 = (_DWORD *)v4[2];
          if ( v31 )
            --*v31;
          if ( v101 )
            j_j___libc_free_0((unsigned __int64)v101);
          if ( v92 )
            j_j___libc_free_0((unsigned __int64)v92);
          nullsub_61();
          v126 = &unk_49DA100;
          nullsub_63();
          if ( v111 != (unsigned int *)v113 )
            _libc_free((unsigned __int64)v111);
          if ( v93 )
            j_j___libc_free_0((unsigned __int64)v93);
        }
      }
    }
    v4 += 3;
    if ( v95 == v4 )
      break;
  }
  for ( k = v108; k != &v106; k = (__int64 *)sub_220EEE0((__int64)k) )
  {
    sub_BD84D0(k[4], k[5]);
    sub_B43D60((_QWORD *)k[4]);
  }
  v33 = (unsigned __int64)v107;
LABEL_47:
  sub_26F73E0(v33);
}
