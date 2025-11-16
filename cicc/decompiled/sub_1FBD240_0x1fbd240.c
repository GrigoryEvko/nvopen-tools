// Function: sub_1FBD240
// Address: 0x1fbd240
//
__int64 *__fastcall sub_1FBD240(
        _QWORD *a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128 a7,
        __m128i a8,
        __m128i a9)
{
  __int64 v9; // r15
  unsigned int v10; // r14d
  unsigned __int64 v11; // r12
  __int64 v13; // rdx
  __int16 v14; // ax
  __int64 *result; // rax
  __int64 v16; // rsi
  unsigned int v17; // edi
  unsigned __int64 *v18; // r13
  unsigned __int64 v19; // r10
  __int64 v20; // rax
  __int64 v21; // rax
  unsigned int v22; // edx
  __int64 *v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int64 v26; // rax
  int v27; // esi
  __int64 v28; // rax
  __int64 v29; // rsi
  unsigned int v30; // r12d
  int v31; // eax
  __int64 *v32; // r15
  __int64 v33; // r14
  __int64 v34; // r12
  __int64 v35; // rax
  __int64 v36; // rbx
  __int64 v37; // rdx
  __int64 v38; // r13
  unsigned __int8 *v39; // r14
  __int64 v40; // r8
  __int64 v41; // rcx
  unsigned __int8 (__fastcall *v42)(__int64, __int64, __int64, __int64, __int64); // r14
  __int64 v43; // rax
  const void **v44; // rdx
  const void **v45; // r14
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rax
  unsigned __int64 v51; // r12
  __int16 *v52; // r13
  bool v53; // al
  __int64 v54; // r8
  __int64 v55; // r11
  __int64 v56; // r9
  bool v57; // r10
  __int64 v58; // r9
  __int64 v59; // rax
  unsigned __int8 v60; // r14
  __int64 v61; // rcx
  __int64 v62; // rax
  __int64 *v63; // rax
  __int64 v64; // rsi
  __int64 *v65; // r15
  unsigned int v66; // ebx
  __int64 v67; // rdx
  __int64 v68; // rax
  __int64 v69; // rdx
  __int64 v70; // r15
  __int64 (__fastcall *v71)(__int64, __int64, __int64, __int64, const void **); // r14
  __int64 v72; // rax
  unsigned __int8 v73; // al
  const void **v74; // rdx
  __int64 v75; // rdi
  bool v76; // al
  int v77; // eax
  int v78; // eax
  int v79; // eax
  __int128 v80; // [rsp-30h] [rbp-140h]
  __int64 v81; // [rsp+0h] [rbp-110h]
  __int64 v82; // [rsp+8h] [rbp-108h]
  __int16 *v83; // [rsp+10h] [rbp-100h]
  __int64 v84; // [rsp+10h] [rbp-100h]
  bool v85; // [rsp+10h] [rbp-100h]
  unsigned __int64 v86; // [rsp+18h] [rbp-F8h]
  int v87; // [rsp+18h] [rbp-F8h]
  __int64 v88; // [rsp+18h] [rbp-F8h]
  __int64 v89; // [rsp+18h] [rbp-F8h]
  unsigned __int64 v90; // [rsp+18h] [rbp-F8h]
  __int64 v91; // [rsp+20h] [rbp-F0h]
  unsigned int v92; // [rsp+20h] [rbp-F0h]
  __int128 v93; // [rsp+20h] [rbp-F0h]
  unsigned __int64 v94; // [rsp+20h] [rbp-F0h]
  __int64 *v95; // [rsp+30h] [rbp-E0h]
  unsigned __int64 v96; // [rsp+30h] [rbp-E0h]
  unsigned __int64 v97; // [rsp+30h] [rbp-E0h]
  __int64 v98; // [rsp+30h] [rbp-E0h]
  __int64 v99; // [rsp+30h] [rbp-E0h]
  const void **v100; // [rsp+30h] [rbp-E0h]
  __int64 *v101; // [rsp+30h] [rbp-E0h]
  __int64 v102; // [rsp+30h] [rbp-E0h]
  __int64 v103; // [rsp+30h] [rbp-E0h]
  __int64 v104; // [rsp+30h] [rbp-E0h]
  __int64 v105; // [rsp+40h] [rbp-D0h]
  __int64 v106; // [rsp+40h] [rbp-D0h]
  __int64 v107; // [rsp+40h] [rbp-D0h]
  __int64 v108; // [rsp+40h] [rbp-D0h]
  unsigned int v109; // [rsp+40h] [rbp-D0h]
  unsigned int v110; // [rsp+40h] [rbp-D0h]
  int v111; // [rsp+48h] [rbp-C8h]
  __int64 *v112; // [rsp+48h] [rbp-C8h]
  __int64 v113; // [rsp+48h] [rbp-C8h]
  __int64 v114; // [rsp+50h] [rbp-C0h] BYREF
  int v115; // [rsp+58h] [rbp-B8h]
  __int64 v116; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v117; // [rsp+68h] [rbp-A8h]
  __int64 v118; // [rsp+70h] [rbp-A0h]
  __int64 v119; // [rsp+78h] [rbp-98h]
  __int64 *v120; // [rsp+80h] [rbp-90h]
  __int64 v121; // [rsp+88h] [rbp-88h]
  __int64 v122; // [rsp+90h] [rbp-80h]
  __int64 v123; // [rsp+98h] [rbp-78h]
  int v124; // [rsp+A0h] [rbp-70h]
  __int64 v125; // [rsp+A8h] [rbp-68h]
  int v126; // [rsp+B0h] [rbp-60h]
  __int64 v127; // [rsp+B8h] [rbp-58h] BYREF
  unsigned int v128; // [rsp+C0h] [rbp-50h]
  __int64 *v129; // [rsp+C8h] [rbp-48h]
  __int64 v130; // [rsp+D0h] [rbp-40h]
  __int64 v131; // [rsp+D8h] [rbp-38h] BYREF

  v9 = a2;
  v10 = a3;
  v11 = a3;
  v13 = *(unsigned __int16 *)(a2 + 24);
  if ( (_WORD)v13 != 124 )
  {
    v14 = *(_WORD *)(a2 + 24);
    if ( (_DWORD)v13 != 145 )
      goto LABEL_3;
    if ( !sub_1D18C00(**(_QWORD **)(a2 + 32), 1, *(_DWORD *)(*(_QWORD *)(a2 + 32) + 8LL)) )
      goto LABEL_11;
    v13 = *(_QWORD *)(a2 + 32);
    v14 = *(_WORD *)(a2 + 24);
    v16 = *(_QWORD *)v13;
    if ( *(_WORD *)(*(_QWORD *)v13 + 24LL) != 124 )
    {
LABEL_3:
      if ( v14 != 120 )
        return 0;
LABEL_12:
      v124 = 0;
      v121 = sub_1D274F0(1u, v13, a4, a5, a6);
      v123 = 0x100000000LL;
      v131 = 0;
      v116 = 0;
      v117 = 0;
      v118 = 0;
      v119 = -4294967084LL;
      v122 = 0;
      v130 = 0;
      v125 = 0;
      v126 = -65536;
      v129 = &v116;
      v127 = v9;
      v128 = v10;
      v20 = *(_QWORD *)(v9 + 48);
      v131 = v20;
      if ( v20 )
        *(_QWORD *)(v20 + 24) = &v131;
      v130 = v9 + 48;
      *(_QWORD *)(v9 + 48) = &v127;
      LODWORD(v123) = 1;
      v120 = &v127;
      if ( *(_WORD *)(v9 + 24) != 120 )
        goto LABEL_21;
      while ( 1 )
      {
        v21 = sub_1FBB600((__int64)a1, v9, *(double *)a7.m128_u64, a8);
        if ( !v21 )
          break;
        if ( v9 == v21 )
        {
          v9 = v127;
          v10 = v128;
          v11 = v128 | v11 & 0xFFFFFFFF00000000LL;
        }
        else
        {
          v10 = v22;
          v9 = v21;
        }
        if ( *(_WORD *)(v9 + 24) != 120 )
          goto LABEL_21;
      }
      if ( *(_WORD *)(v9 + 24) == 120 )
      {
        v50 = *(_QWORD *)(v9 + 32);
        v51 = *(_QWORD *)v50;
        v52 = *(__int16 **)(v50 + 8);
        v99 = *(_QWORD *)v50;
        if ( *(_WORD *)(*(_QWORD *)v50 + 24LL) == 137 )
          goto LABEL_39;
        v106 = *(_QWORD *)(v50 + 40);
        if ( *(_WORD *)(v106 + 24) == 137 )
          goto LABEL_39;
        v87 = *(_DWORD *)(v50 + 8);
        v92 = *(_DWORD *)(v50 + 48);
        v53 = sub_1D18910(v51);
        v55 = v106;
        v56 = v99;
        v57 = v53;
        if ( !v53 )
          goto LABEL_41;
        v75 = v99;
        v102 = v106;
        v108 = v56;
        v76 = sub_1D18C00(v75, 1, v87);
        v55 = v102;
        v57 = v76;
        if ( v76 )
        {
          v58 = v108;
          if ( *(_WORD *)(v108 + 24) != 120 )
          {
            v58 = v9;
            v57 = 0;
          }
        }
        else
        {
LABEL_41:
          v58 = v9;
        }
        v59 = *(_QWORD *)(v9 + 40) + 16LL * v10;
        v60 = *(_BYTE *)v59;
        v100 = *(const void ***)(v59 + 8);
        if ( *((_BYTE *)a1 + 25) )
        {
          v70 = a1[1];
          v81 = v55;
          v82 = v58;
          v85 = v57;
          v89 = v60;
          v71 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, const void **))(*(_QWORD *)v70 + 264LL);
          v107 = *(_QWORD *)(*a1 + 48LL);
          v72 = sub_1E0A0C0(*(_QWORD *)(*a1 + 32LL));
          v73 = v71(v70, v72, v107, v89, v100);
          v55 = v81;
          v58 = v82;
          v100 = v74;
          v57 = v85;
          v60 = v73;
        }
        if ( (v61 = a1[1], v62 = 1, v60 != 1) && (!v60 || (v62 = v60, !*(_QWORD *)(v61 + 8LL * v60 + 120)))
          || (*(_BYTE *)(v61 + 259 * v62 + 2614) & 0xFB) != 0 )
        {
LABEL_39:
          sub_1D189A0((__int64)&v116);
          return 0;
        }
        v63 = (__int64 *)*a1;
        v64 = *(_QWORD *)(v58 + 72);
        v114 = v64;
        v65 = v63;
        v66 = !v57 ? 22 : 17;
        if ( v64 )
        {
          v84 = v55;
          v88 = v58;
          sub_1623A60((__int64)&v114, v64, 2);
          v55 = v84;
          v58 = v88;
        }
        v115 = *(_DWORD *)(v58 + 64);
        v67 = v92;
        *(_QWORD *)&v93 = v55;
        *((_QWORD *)&v93 + 1) = v67;
        v68 = sub_1D28D50(v65, v66, v67, v60, v54, 0);
        v23 = sub_1D3A900(
                v65,
                0x89u,
                (__int64)&v114,
                v60,
                v100,
                0,
                a7,
                *(double *)a8.m128i_i64,
                a9,
                v51,
                v52,
                v93,
                v68,
                v69);
        if ( v114 )
        {
          v101 = v23;
          sub_161E7C0((__int64)&v114, v114);
          v23 = v101;
        }
      }
      else
      {
LABEL_21:
        v23 = (__int64 *)v9;
      }
      v95 = v23;
      sub_1D189A0((__int64)&v116);
      return v95;
    }
    v17 = *(_DWORD *)(v13 + 8);
    if ( v14 != 145 )
      goto LABEL_11;
    v13 = 0xFFFFFFFF00000000LL;
    v10 = v17;
    v9 = v16;
    v11 = v17 | v11 & 0xFFFFFFFF00000000LL;
  }
  v18 = *(unsigned __int64 **)(v9 + 32);
  v19 = *v18;
  if ( *(_WORD *)(*v18 + 24) != 118 )
    goto LABEL_11;
  a4 = v18[5];
  if ( *(_WORD *)(a4 + 24) != 10 )
    goto LABEL_11;
  v24 = *(_QWORD *)(*(_QWORD *)(v19 + 32) + 40LL);
  if ( *(_WORD *)(v24 + 24) != 10 )
    goto LABEL_11;
  v25 = *(_QWORD *)(v24 + 88);
  v13 = *(unsigned int *)(v25 + 32);
  if ( (unsigned int)v13 > 0x40 )
  {
    v94 = *v18;
    v109 = *(_DWORD *)(v25 + 32);
    v103 = v18[5];
    v113 = v25 + 24;
    v77 = sub_16A5940(v25 + 24);
    a4 = v103;
    v13 = v109;
    if ( v77 != 1 )
      goto LABEL_11;
    v78 = sub_16A57B0(v113);
    v19 = v94;
    LODWORD(v13) = v109;
    a4 = v103;
    v27 = v78;
  }
  else
  {
    v26 = *(_QWORD *)(v25 + 24);
    if ( !v26 || (v26 & (v26 - 1)) != 0 )
      goto LABEL_11;
    _BitScanReverse64(&v26, v26);
    v27 = v13 + (v26 ^ 0x3F) - 64;
  }
  a4 = *(_QWORD *)(a4 + 88);
  a5 = *(unsigned int *)(a4 + 32);
  v111 = *(_DWORD *)(a4 + 32);
  if ( (unsigned int)a5 <= 0x40 )
  {
    v28 = *(_QWORD *)(a4 + 24);
    goto LABEL_31;
  }
  v90 = v19;
  v110 = v13;
  v104 = a4;
  v79 = sub_16A57B0(a4 + 24);
  a4 = v104;
  v13 = v110;
  a5 = (unsigned int)(v111 - v79);
  v19 = v90;
  if ( (unsigned int)a5 > 0x40 )
  {
LABEL_11:
    if ( *(_WORD *)(v9 + 24) != 120 )
      return 0;
    goto LABEL_12;
  }
  v28 = **(_QWORD **)(v104 + 24);
LABEL_31:
  v13 = (unsigned int)(v13 - 1 - v27);
  if ( v13 != v28 )
    goto LABEL_11;
  v29 = *(_QWORD *)(v9 + 72);
  v30 = *((_DWORD *)v18 + 2);
  v116 = v29;
  if ( v29 )
  {
    v96 = v19;
    sub_1623A60((__int64)&v116, v29, 2);
    v19 = v96;
  }
  v31 = *(_DWORD *)(v9 + 64);
  v32 = (__int64 *)*a1;
  LODWORD(v117) = v31;
  v33 = 16LL * v30;
  v83 = (__int16 *)v30;
  v97 = v19;
  v34 = sub_1D38BB0(
          (__int64)v32,
          0,
          (__int64)&v116,
          *(unsigned __int8 *)(v33 + *(_QWORD *)(v19 + 40)),
          *(const void ***)(v33 + *(_QWORD *)(v19 + 40) + 8),
          0,
          (__m128i)a7,
          *(double *)a8.m128i_i64,
          a9,
          0);
  v35 = *a1;
  v36 = a1[1];
  v38 = v37;
  v39 = (unsigned __int8 *)(*(_QWORD *)(v97 + 40) + v33);
  v86 = v97;
  v40 = *((_QWORD *)v39 + 1);
  v41 = *v39;
  v42 = *(unsigned __int8 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v36 + 264LL);
  v91 = v40;
  v105 = v41;
  v98 = *(_QWORD *)(v35 + 48);
  v43 = sub_1E0A0C0(*(_QWORD *)(v35 + 32));
  LODWORD(v36) = v42(v36, v43, v98, v105, v91);
  v45 = v44;
  v48 = sub_1D28D50(v32, 0x16u, 0xFFFFFFFF00000000LL, v46, v47, 0);
  *((_QWORD *)&v80 + 1) = v38;
  *(_QWORD *)&v80 = v34;
  result = sub_1D3A900(
             v32,
             0x89u,
             (__int64)&v116,
             (unsigned int)v36,
             v45,
             0,
             a7,
             *(double *)a8.m128i_i64,
             a9,
             v86,
             v83,
             v80,
             v48,
             v49);
  if ( v116 )
  {
    v112 = result;
    sub_161E7C0((__int64)&v116, v116);
    return v112;
  }
  return result;
}
