// Function: sub_1A3A670
// Address: 0x1a3a670
//
void __fastcall sub_1A3A670(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // rax
  double v12; // xmm4_8
  double v13; // xmm5_8
  int v14; // edx
  unsigned __int64 v15; // rdi
  __int64 v16; // rax
  __int64 *v17; // rax
  _BYTE *v18; // rax
  __int64 **v19; // rbx
  _QWORD *v20; // r8
  int v21; // esi
  unsigned int v22; // ecx
  _QWORD *v23; // rax
  __int64 v24; // r9
  __int64 v25; // rsi
  __int64 *v26; // r12
  __int64 v27; // rdx
  unsigned int v28; // esi
  unsigned int v29; // eax
  unsigned int v30; // ecx
  unsigned int v31; // r8d
  __m128i v32; // rax
  int v33; // r13d
  int v34; // r13d
  __int64 v35; // rax
  __int64 v36; // r15
  __int64 v37; // rbx
  __int64 v38; // rdx
  unsigned __int64 v39; // r15
  __int64 v40; // rax
  __int64 v41; // rsi
  _QWORD *v42; // rdi
  int v43; // esi
  __int64 v44; // rax
  __int64 *v45; // r8
  __int64 v46; // r9
  __int64 v47; // rdx
  __int64 v48; // rdx
  _QWORD *v49; // rax
  _QWORD *v50; // r15
  _QWORD *v51; // r15
  __int64 v52; // rcx
  __int64 v53; // r8
  __int64 v54; // r9
  __int64 v55; // rdi
  __int64 v56; // rdx
  __int64 v57; // rsi
  __int64 v58; // rdx
  int v59; // eax
  __int64 v60; // rax
  int v61; // ecx
  __int64 v62; // rcx
  _QWORD *v63; // rax
  __int64 v64; // rdi
  unsigned __int64 v65; // rcx
  __int64 v66; // rcx
  __int64 v67; // rcx
  __int64 v68; // rax
  __int64 v69; // rcx
  _BYTE *v70; // rbx
  _QWORD *v71; // r12
  int v72; // r10d
  unsigned int v73; // esi
  unsigned int v74; // eax
  unsigned int v75; // ecx
  unsigned int v76; // edi
  __int64 v77; // rax
  __int64 *v78; // rax
  int v79; // r11d
  __int64 *v80; // r10
  __int64 **v81; // [rsp-430h] [rbp-430h]
  __int64 v82; // [rsp-428h] [rbp-428h]
  __int64 v83; // [rsp-3E0h] [rbp-3E0h]
  __int64 v84; // [rsp-3D8h] [rbp-3D8h]
  __int64 **v85; // [rsp-3C0h] [rbp-3C0h]
  __int64 v86; // [rsp-3B8h] [rbp-3B8h]
  __int64 v87; // [rsp-3B8h] [rbp-3B8h]
  __int64 v88; // [rsp-3B0h] [rbp-3B0h]
  __int64 v89; // [rsp-3A8h] [rbp-3A8h]
  __int64 *v90; // [rsp-3A0h] [rbp-3A0h]
  _QWORD *v91; // [rsp-3A0h] [rbp-3A0h]
  __int64 v92; // [rsp-398h] [rbp-398h] BYREF
  __int64 v93; // [rsp-390h] [rbp-390h] BYREF
  _QWORD v94[2]; // [rsp-388h] [rbp-388h] BYREF
  _QWORD v95[2]; // [rsp-378h] [rbp-378h] BYREF
  __m128i v96; // [rsp-368h] [rbp-368h] BYREF
  __int16 v97; // [rsp-358h] [rbp-358h]
  __m128i v98; // [rsp-348h] [rbp-348h] BYREF
  __int16 v99; // [rsp-338h] [rbp-338h]
  __m128i v100; // [rsp-328h] [rbp-328h] BYREF
  __int16 v101; // [rsp-318h] [rbp-318h]
  __int64 **v102; // [rsp-308h] [rbp-308h] BYREF
  __int64 v103; // [rsp-300h] [rbp-300h]
  _BYTE v104[32]; // [rsp-2F8h] [rbp-2F8h] BYREF
  __int64 v105; // [rsp-2D8h] [rbp-2D8h] BYREF
  __int64 v106; // [rsp-2D0h] [rbp-2D0h]
  _QWORD *v107; // [rsp-2C8h] [rbp-2C8h] BYREF
  unsigned int v108; // [rsp-2C0h] [rbp-2C0h]
  __int64 v109[8]; // [rsp-288h] [rbp-288h] BYREF
  __int64 *v110; // [rsp-248h] [rbp-248h]
  __int64 v111; // [rsp-238h] [rbp-238h] BYREF
  __int64 v112[2]; // [rsp-218h] [rbp-218h] BYREF
  unsigned __int64 v113; // [rsp-208h] [rbp-208h]
  __int64 v114; // [rsp-200h] [rbp-200h]
  __int64 v115; // [rsp-1F8h] [rbp-1F8h]
  int v116; // [rsp-1F0h] [rbp-1F0h]
  __int64 v117; // [rsp-1E8h] [rbp-1E8h]
  __int64 v118; // [rsp-1E0h] [rbp-1E0h]
  _QWORD *v119; // [rsp-1D8h] [rbp-1D8h]
  __int64 v120; // [rsp-1D0h] [rbp-1D0h]
  _QWORD v121[4]; // [rsp-1C8h] [rbp-1C8h] BYREF
  __int64 v122; // [rsp-1A8h] [rbp-1A8h] BYREF
  __int64 v123; // [rsp-1A0h] [rbp-1A0h]
  _QWORD *v124; // [rsp-198h] [rbp-198h] BYREF
  unsigned int v125; // [rsp-190h] [rbp-190h]
  _BYTE v126[56]; // [rsp-38h] [rbp-38h] BYREF

  if ( !a2 )
    return;
  v11 = sub_15F2050(a1);
  sub_1632FA0(v11);
  v102 = (__int64 **)v104;
  v103 = 0x400000000LL;
  sub_1A24B00(a1, (__int64)&v102);
  v14 = v103;
  if ( !(_DWORD)v103 )
  {
    v15 = (unsigned __int64)v102;
    if ( v102 == (__int64 **)v104 )
      return;
    goto LABEL_4;
  }
  v16 = *(_QWORD *)(a1 + 40);
  v105 = 0;
  v106 = 1;
  v84 = v16;
  v17 = (__int64 *)&v107;
  do
  {
    *v17 = -8;
    v17 += 2;
  }
  while ( v17 != v109 );
  v18 = &v124;
  v122 = 0;
  v123 = 1;
  do
  {
    *(_QWORD *)v18 = -8;
    v18 += 88;
  }
  while ( v18 != v126 );
  v19 = v102;
  v85 = &v102[v14];
  do
  {
    v26 = *v19;
    v27 = **v19;
    v92 = v27;
    v89 = *(v26 - 3);
    if ( (v106 & 1) != 0 )
    {
      v20 = &v107;
      v21 = 3;
    }
    else
    {
      v28 = v108;
      v20 = v107;
      if ( !v108 )
      {
        v29 = v106;
        ++v105;
        v90 = 0;
        v30 = ((unsigned int)v106 >> 1) + 1;
        goto LABEL_19;
      }
      v21 = v108 - 1;
    }
    v22 = v21 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
    v23 = &v20[2 * v22];
    v24 = *v23;
    if ( v27 != *v23 )
    {
      v91 = 0;
      v72 = 1;
      while ( v24 != -8 )
      {
        if ( !v91 )
        {
          if ( v24 != -16 )
            v23 = v91;
          v91 = v23;
        }
        v22 = v21 & (v72 + v22);
        v23 = &v20[2 * v22];
        v24 = *v23;
        if ( v27 == *v23 )
          goto LABEL_13;
        ++v72;
      }
      v31 = 12;
      if ( v91 )
        v23 = v91;
      ++v105;
      v28 = 4;
      v90 = v23;
      v29 = v106;
      v30 = ((unsigned int)v106 >> 1) + 1;
      if ( (v106 & 1) != 0 )
      {
LABEL_20:
        if ( 4 * v30 >= v31 )
        {
          v28 *= 2;
        }
        else if ( v28 - HIDWORD(v106) - v30 > v28 >> 3 )
        {
LABEL_22:
          LODWORD(v106) = (2 * (v29 >> 1) + 2) | v29 & 1;
          if ( *v90 != -8 )
            --HIDWORD(v106);
          *v90 = v27;
          v90[1] = 0;
          goto LABEL_25;
        }
        sub_1A2BB90((__int64)&v105, v28);
        sub_1A27140((__int64)&v105, &v92, v112);
        v27 = v92;
        v90 = (__int64 *)v112[0];
        v29 = v106;
        goto LABEL_22;
      }
      v28 = v108;
LABEL_19:
      v31 = 3 * v28;
      goto LABEL_20;
    }
LABEL_13:
    v25 = v23[1];
    if ( v25 )
      goto LABEL_14;
    v90 = v23;
LABEL_25:
    sub_1A1B630((__int64)v109, a1);
    v32.m128i_i64[0] = (__int64)sub_1649960(a1);
    v33 = *(_DWORD *)(a1 + 20);
    v98 = v32;
    v88 = v92;
    v34 = v33 & 0xFFFFFFF;
    v100.m128i_i64[0] = (__int64)&v98;
    v101 = 773;
    v100.m128i_i64[1] = (__int64)".sroa.speculated";
    LOWORD(v113) = 257;
    v35 = sub_1648B60(64);
    v36 = v35;
    if ( v35 )
    {
      sub_15F1EA0(v35, v88, 53, 0, 0, 0);
      *(_DWORD *)(v36 + 56) = v34;
      sub_164B780(v36, v112);
      sub_1648880(v36, *(_DWORD *)(v36 + 56), 1);
    }
    v90[1] = (__int64)sub_1A1C7B0(v109, (_QWORD *)v36, &v100);
    if ( (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) != 0 )
    {
      v81 = v19;
      v37 = 0;
      v83 = 8LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      while ( 1 )
      {
        if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
          v38 = *(_QWORD *)(a1 - 8);
        else
          v38 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
        v93 = *(_QWORD *)(v37 + v38 + 24LL * *(unsigned int *)(a1 + 56) + 8);
        v39 = sub_157EBA0(v93);
        v40 = sub_16498A0(v39);
        LOBYTE(v121[0]) = 0;
        v114 = v40;
        v112[0] = 0;
        v115 = 0;
        v116 = 0;
        v117 = 0;
        v118 = 0;
        v119 = v121;
        v120 = 0;
        v112[1] = *(_QWORD *)(v39 + 40);
        v113 = v39 + 24;
        v41 = *(_QWORD *)(v39 + 48);
        v100.m128i_i64[0] = v41;
        if ( v41 )
        {
          sub_1623A60((__int64)&v100, v41, 2);
          if ( v112[0] )
            sub_161E7C0((__int64)v112, v112[0]);
          v112[0] = v100.m128i_i64[0];
          if ( v100.m128i_i64[0] )
            sub_1623210((__int64)&v100, (unsigned __int8 *)v100.m128i_i64[0], (__int64)v112);
        }
        if ( (v123 & 1) != 0 )
        {
          v42 = &v124;
          v43 = 3;
        }
        else
        {
          v73 = v125;
          v42 = v124;
          if ( !v125 )
          {
            v74 = v123;
            ++v122;
            v45 = 0;
            v75 = ((unsigned int)v123 >> 1) + 1;
            goto LABEL_96;
          }
          v43 = v125 - 1;
        }
        v44 = v43 & (((unsigned int)v93 >> 9) ^ ((unsigned int)v93 >> 4));
        v45 = &v42[11 * v44];
        v46 = *v45;
        if ( v93 != *v45 )
          break;
LABEL_38:
        v86 = sub_1A2A4F0(v89, v112, v84, v46, (__int64)(v45 + 1));
        v95[0] = sub_1649960(v93);
        v99 = 261;
        v95[1] = v47;
        v98.m128i_i64[0] = (__int64)v95;
        v94[0] = sub_1649960(a1);
        v94[1] = v48;
        v97 = 773;
        v96.m128i_i64[0] = (__int64)v94;
        v96.m128i_i64[1] = (__int64)".sroa.speculate.load.";
        sub_14EC200(&v100, &v96, &v98);
        v49 = sub_1648A60(64, 1u);
        v50 = v49;
        if ( v49 )
          sub_15F9210((__int64)v49, *(_QWORD *)(*(_QWORD *)v86 + 24LL), v86, 0, 0, 0);
        v51 = sub_1A1C7B0(v112, v50, &v100);
        sub_15F8F50((__int64)v51, 1 << (*((unsigned __int16 *)v26 + 9) >> 1) >> 1);
        v55 = (__int64)*v102;
        if ( (*v102)[6] || *(__int16 *)(v55 + 18) < 0 )
        {
          v56 = sub_1625790(v55, 1);
          if ( v56 )
            sub_1625C10((__int64)v51, 1, v56);
        }
        v57 = v93;
        v58 = v90[1];
        v59 = *(_DWORD *)(v58 + 20) & 0xFFFFFFF;
        if ( v59 == *(_DWORD *)(v58 + 56) )
        {
          v82 = v93;
          v87 = v90[1];
          sub_15F55D0(v87, v93, v58, v52, v53, v54);
          v58 = v87;
          v57 = v82;
          v59 = *(_DWORD *)(v87 + 20) & 0xFFFFFFF;
        }
        v60 = (v59 + 1) & 0xFFFFFFF;
        v61 = v60 | *(_DWORD *)(v58 + 20) & 0xF0000000;
        *(_DWORD *)(v58 + 20) = v61;
        if ( (v61 & 0x40000000) != 0 )
          v62 = *(_QWORD *)(v58 - 8);
        else
          v62 = v58 - 24 * v60;
        v63 = (_QWORD *)(v62 + 24LL * (unsigned int)(v60 - 1));
        if ( *v63 )
        {
          v64 = v63[1];
          v65 = v63[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v65 = v64;
          if ( v64 )
            *(_QWORD *)(v64 + 16) = *(_QWORD *)(v64 + 16) & 3LL | v65;
        }
        *v63 = v51;
        if ( v51 )
        {
          v66 = v51[1];
          v63[1] = v66;
          if ( v66 )
            *(_QWORD *)(v66 + 16) = (unsigned __int64)(v63 + 1) | *(_QWORD *)(v66 + 16) & 3LL;
          v63[2] = (unsigned __int64)(v51 + 1) | v63[2] & 3LL;
          v51[1] = v63;
        }
        v67 = *(_DWORD *)(v58 + 20) & 0xFFFFFFF;
        v68 = (unsigned int)(v67 - 1);
        if ( (*(_BYTE *)(v58 + 23) & 0x40) != 0 )
          v69 = *(_QWORD *)(v58 - 8);
        else
          v69 = v58 - 24 * v67;
        *(_QWORD *)(v69 + 8 * v68 + 24LL * *(unsigned int *)(v58 + 56) + 8) = v57;
        if ( v119 != v121 )
          j_j___libc_free_0(v119, v121[0] + 1LL);
        if ( v112[0] )
          sub_161E7C0((__int64)v112, v112[0]);
        v37 += 8;
        if ( v83 == v37 )
        {
          v19 = v81;
          goto LABEL_107;
        }
      }
      v79 = 1;
      v80 = 0;
      while ( v46 != -8 )
      {
        if ( v46 == -16 && !v80 )
          v80 = v45;
        LODWORD(v44) = v43 & (v79 + v44);
        v45 = &v42[11 * (unsigned int)v44];
        v46 = *v45;
        if ( v93 == *v45 )
          goto LABEL_38;
        ++v79;
      }
      v74 = v123;
      v76 = 12;
      v73 = 4;
      if ( v80 )
        v45 = v80;
      ++v122;
      v75 = ((unsigned int)v123 >> 1) + 1;
      if ( (v123 & 1) == 0 )
      {
        v73 = v125;
LABEL_96:
        v76 = 3 * v73;
      }
      if ( v76 <= 4 * v75 )
      {
        v73 *= 2;
      }
      else if ( v73 - HIDWORD(v123) - v75 > v73 >> 3 )
      {
LABEL_99:
        LODWORD(v123) = (2 * (v74 >> 1) + 2) | v74 & 1;
        if ( *v45 != -8 )
          --HIDWORD(v123);
        v77 = v93;
        v45[1] = 0;
        v45[2] = 1;
        *v45 = v77;
        v78 = v45 + 3;
        do
        {
          if ( v78 )
            *v78 = -8;
          v78 += 2;
        }
        while ( v78 != v45 + 11 );
        v46 = v93;
        goto LABEL_38;
      }
      sub_1A3A300((__int64)&v122, v73);
      sub_1A27200((__int64)&v122, &v93, &v100);
      v45 = (__int64 *)v100.m128i_i64[0];
      v74 = v123;
      goto LABEL_99;
    }
LABEL_107:
    if ( v110 != &v111 )
      j_j___libc_free_0(v110, v111 + 1);
    if ( v109[0] )
      sub_161E7C0((__int64)v109, v109[0]);
    v25 = v90[1];
LABEL_14:
    ++v19;
    sub_164D160((__int64)v26, v25, a3, a4, a5, a6, v12, v13, a9, a10);
    sub_15F20C0(v26);
  }
  while ( v85 != v19 );
  v70 = v126;
  v71 = &v124;
  if ( (v123 & 1) == 0 )
  {
    v71 = v124;
    if ( !v125 )
      goto LABEL_87;
    v70 = &v124[11 * v125];
  }
  do
  {
    if ( *v71 != -16 && *v71 != -8 && (v71[2] & 1) == 0 )
      j___libc_free_0(v71[3]);
    v71 += 11;
  }
  while ( v70 != (_BYTE *)v71 );
  if ( (v123 & 1) == 0 )
  {
    v71 = v124;
LABEL_87:
    j___libc_free_0(v71);
    if ( (v106 & 1) != 0 )
    {
LABEL_73:
      v15 = (unsigned __int64)v102;
      if ( v102 == (__int64 **)v104 )
        return;
LABEL_4:
      _libc_free(v15);
      return;
    }
    goto LABEL_84;
  }
  if ( (v106 & 1) != 0 )
    goto LABEL_73;
LABEL_84:
  j___libc_free_0(v107);
  v15 = (unsigned __int64)v102;
  if ( v102 != (__int64 **)v104 )
    goto LABEL_4;
}
