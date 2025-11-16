// Function: sub_2CCDD70
// Address: 0x2ccdd70
//
void __fastcall sub_2CCDD70(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        unsigned int a5,
        char a6,
        _QWORD *a7,
        __int64 a8)
{
  unsigned int v9; // eax
  unsigned int v10; // ecx
  unsigned int v11; // ebx
  bool v12; // r14
  unsigned int v13; // eax
  __int64 v14; // rdi
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // rax
  __int64 v17; // r13
  int i; // edx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 **v21; // r15
  __int64 (__fastcall *v22)(__int64, unsigned int, _BYTE *, __int64); // rax
  unsigned __int8 *v23; // r13
  __int64 (__fastcall *v24)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  __int64 v25; // rbx
  __int64 v26; // rax
  int v27; // r15d
  __int64 v28; // rdx
  unsigned __int64 *v29; // rax
  unsigned __int64 v30; // r12
  __int64 v31; // rax
  __int64 v32; // r14
  _QWORD *v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r10
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  _QWORD *v39; // r12
  __int64 v40; // r11
  unsigned __int64 v41; // rdi
  __int64 v42; // rdx
  int v43; // eax
  char v44; // al
  int v45; // edx
  unsigned __int8 *v46; // rax
  unsigned int *v47; // r15
  unsigned int *v48; // rbx
  __int64 v49; // rdx
  unsigned int v50; // esi
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 **v53; // rbx
  __int64 (__fastcall *v54)(__int64, unsigned int, _BYTE *, __int64); // rax
  _BYTE *v55; // r15
  __int64 v56; // rax
  _QWORD *v57; // rax
  __int64 v58; // r13
  _QWORD *v59; // rsi
  __int64 v60; // r8
  __int64 v61; // r9
  __int64 v62; // r12
  __int64 v63; // rax
  unsigned __int64 v64; // rdx
  __int64 *v65; // r12
  _QWORD *v66; // r13
  __int64 v67; // r11
  unsigned int v68; // ecx
  __int64 *v69; // rax
  __int64 v70; // r14
  __int64 v71; // rax
  __int64 v72; // r15
  _QWORD *v73; // rax
  __int64 v74; // r12
  __int64 v75; // r13
  char v76; // r8
  unsigned __int64 v77; // rax
  _QWORD *v78; // rdi
  __int64 v79; // r9
  __int64 v80; // rax
  unsigned int *v81; // r15
  unsigned int *v82; // r13
  __int64 v83; // rdx
  unsigned int v84; // esi
  __int64 v85; // r12
  __int64 v86; // rax
  __int64 *v87; // rdi
  __int64 *v88; // rax
  __int64 v89; // rsi
  int v90; // edx
  char v91; // al
  int v92; // edx
  __int64 v93; // rax
  unsigned int *v94; // r12
  unsigned int *v95; // rbx
  __int64 v96; // rdx
  unsigned int v97; // esi
  int v98; // [rsp+4h] [rbp-1BCh]
  unsigned int v99; // [rsp+4h] [rbp-1BCh]
  unsigned __int64 v100; // [rsp+8h] [rbp-1B8h]
  __int64 v101; // [rsp+10h] [rbp-1B0h]
  __int64 v102; // [rsp+18h] [rbp-1A8h]
  _QWORD *v103; // [rsp+18h] [rbp-1A8h]
  __int64 v104; // [rsp+18h] [rbp-1A8h]
  __int64 v105; // [rsp+18h] [rbp-1A8h]
  __int64 v107; // [rsp+28h] [rbp-198h]
  _BYTE *v108; // [rsp+28h] [rbp-198h]
  __int64 v109; // [rsp+30h] [rbp-190h]
  __int64 v110; // [rsp+30h] [rbp-190h]
  unsigned __int8 *v111; // [rsp+40h] [rbp-180h]
  __int64 **v112; // [rsp+40h] [rbp-180h]
  __int64 v113; // [rsp+40h] [rbp-180h]
  __int64 v114; // [rsp+40h] [rbp-180h]
  unsigned int v118; // [rsp+68h] [rbp-158h]
  __int64 v119; // [rsp+68h] [rbp-158h]
  char v120; // [rsp+68h] [rbp-158h]
  _QWORD v121[2]; // [rsp+70h] [rbp-150h] BYREF
  unsigned int v122; // [rsp+80h] [rbp-140h]
  __int16 v123; // [rsp+90h] [rbp-130h]
  __int64 v124[4]; // [rsp+A0h] [rbp-120h] BYREF
  char v125; // [rsp+C0h] [rbp-100h]
  char v126; // [rsp+C1h] [rbp-FFh]
  __int64 *v127; // [rsp+D0h] [rbp-F0h] BYREF
  __int64 v128; // [rsp+D8h] [rbp-E8h]
  __int64 v129[2]; // [rsp+E0h] [rbp-E0h] BYREF
  __int16 v130; // [rsp+F0h] [rbp-D0h]
  char *v131; // [rsp+100h] [rbp-C0h] BYREF
  int v132; // [rsp+108h] [rbp-B8h]
  char v133; // [rsp+110h] [rbp-B0h] BYREF
  char v134; // [rsp+120h] [rbp-A0h]
  char v135; // [rsp+121h] [rbp-9Fh]
  __int64 v136; // [rsp+138h] [rbp-88h]
  __int64 v137; // [rsp+140h] [rbp-80h]
  _QWORD *v138; // [rsp+148h] [rbp-78h]
  __int64 v139; // [rsp+150h] [rbp-70h]
  __int64 v140; // [rsp+158h] [rbp-68h]
  void *v141; // [rsp+180h] [rbp-40h]

  v9 = 1;
  if ( a5 <= 1 )
  {
    v80 = sub_BCB2B0(a7);
    sub_2CCA9F0(a1, v80, a2, a3, a4, a6, (__int64)a7, a8);
    return;
  }
  do
  {
    v10 = v9;
    v9 *= 2;
  }
  while ( (unsigned int)qword_5013708 > v9 );
  v118 = (v10 | a5) & -(v10 | a5);
  if ( *(_BYTE *)a4 == 17 )
  {
    v11 = *(_DWORD *)(a4 + 32);
    v12 = v11 <= 0x40 ? *(_QWORD *)(a4 + 24) == 0 : v11 == (unsigned int)sub_C444A0(a4 + 24);
    if ( v12 )
    {
      v13 = v118;
      if ( v118 > 1 )
        goto LABEL_8;
LABEL_30:
      v109 = a3;
      v25 = a4;
      goto LABEL_31;
    }
  }
  v12 = 0;
  v118 = (v118 | 0x10) & -(v118 | 0x10);
  v13 = v118;
  if ( v118 <= 1 )
    goto LABEL_30;
LABEL_8:
  v14 = *(_QWORD *)(a3 + 8);
  v15 = v13;
  if ( *(_BYTE *)a3 == 17 )
  {
    v16 = *(_QWORD *)(a3 + 24);
    if ( *(_DWORD *)(a3 + 32) > 0x40u )
      v16 = *(_QWORD *)v16;
    v109 = sub_AD64C0(v14, (unsigned int)(v16 / v15), 0);
  }
  else
  {
    v135 = 1;
    v134 = 3;
    v131 = "udiv";
    v31 = sub_AD64C0(v14, v13, 0);
    v109 = sub_B504D0(19, a3, v31, (__int64)&v131, a1 + 24, 0);
  }
  sub_23D0AB0((__int64)&v131, a1, 0, 0, 0);
  if ( v118 <= 4 )
  {
    v17 = 0;
    for ( i = 0; i != v118; ++i )
      v17 |= (v17 << 8) | 1;
    v19 = sub_BCCE00(v138, 8 * v118);
    v20 = sub_ACD640(v19, v17, 0);
    v122 = v118;
    v124[0] = (__int64)"zext";
    v121[0] = "i8x";
    v123 = 2307;
    v126 = 1;
    v125 = 3;
    v21 = *(__int64 ***)(v20 + 8);
    v111 = (unsigned __int8 *)v20;
    if ( v21 == *(__int64 ***)(a4 + 8) )
    {
      v23 = (unsigned __int8 *)a4;
      goto LABEL_21;
    }
    v22 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v139 + 120LL);
    if ( v22 == sub_920130 )
    {
      if ( *(_BYTE *)a4 > 0x15u )
      {
LABEL_50:
        v130 = 257;
        v46 = (unsigned __int8 *)sub_BD2C40(72, 1u);
        v23 = v46;
        if ( v46 )
          sub_B515B0((__int64)v46, a4, (__int64)v21, (__int64)&v127, 0, 0);
        (*(void (__fastcall **)(__int64, unsigned __int8 *, __int64 *, __int64, __int64))(*(_QWORD *)v140 + 16LL))(
          v140,
          v23,
          v124,
          v136,
          v137);
        v47 = (unsigned int *)v131;
        v48 = (unsigned int *)&v131[16 * v132];
        if ( v131 != (char *)v48 )
        {
          do
          {
            v49 = *((_QWORD *)v47 + 1);
            v50 = *v47;
            v47 += 4;
            sub_B99FD0((__int64)v23, v50, v49);
          }
          while ( v48 != v47 );
        }
LABEL_21:
        v24 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))(*(_QWORD *)v139 + 32LL);
        if ( v24 == sub_9201A0 )
        {
          if ( *v23 > 0x15u || *v111 > 0x15u )
            goto LABEL_86;
          if ( (unsigned __int8)sub_AC47B0(17) )
            v25 = sub_AD5570(17, (__int64)v23, v111, 0, 0);
          else
            v25 = sub_AABE40(0x11u, v23, v111);
        }
        else
        {
          v25 = v24(v139, 17u, v23, v111, 0, 0);
        }
        if ( v25 )
          goto LABEL_27;
LABEL_86:
        v130 = 257;
        v25 = sub_B504D0(17, (__int64)v23, (__int64)v111, (__int64)&v127, 0, 0);
        (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)v140 + 16LL))(
          v140,
          v25,
          v121,
          v136,
          v137);
        v81 = (unsigned int *)v131;
        v82 = (unsigned int *)&v131[16 * v132];
        if ( v131 != (char *)v82 )
        {
          do
          {
            v83 = *((_QWORD *)v81 + 1);
            v84 = *v81;
            v81 += 4;
            sub_B99FD0(v25, v84, v83);
          }
          while ( v82 != v81 );
        }
        goto LABEL_27;
      }
      if ( (unsigned __int8)sub_AC4810(0x27u) )
        v23 = (unsigned __int8 *)sub_ADAB70(39, a4, v21, 0);
      else
        v23 = (unsigned __int8 *)sub_AA93C0(0x27u, a4, (__int64)v21);
    }
    else
    {
      v23 = (unsigned __int8 *)v22(v139, 39u, (_BYTE *)a4, (__int64)v21);
    }
    if ( v23 )
      goto LABEL_21;
    goto LABEL_50;
  }
  v51 = sub_BCB2D0(v138);
  v52 = sub_ACD640(v51, 0x101010101LL, 0);
  v121[0] = "i8x4";
  v123 = 259;
  v126 = 1;
  v124[0] = (__int64)"zext";
  v125 = 3;
  v53 = *(__int64 ***)(v52 + 8);
  v108 = (_BYTE *)v52;
  if ( v53 == *(__int64 ***)(a4 + 8) )
  {
    v55 = (_BYTE *)a4;
    goto LABEL_61;
  }
  v54 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v139 + 120LL);
  if ( v54 == sub_920130 )
  {
    if ( *(_BYTE *)a4 > 0x15u )
    {
LABEL_104:
      v130 = 257;
      v55 = sub_BD2C40(72, 1u);
      if ( v55 )
        sub_B515B0((__int64)v55, a4, (__int64)v53, (__int64)&v127, 0, 0);
      (*(void (__fastcall **)(__int64, _BYTE *, __int64 *, __int64, __int64))(*(_QWORD *)v140 + 16LL))(
        v140,
        v55,
        v124,
        v136,
        v137);
      if ( v131 != &v131[16 * v132] )
      {
        v105 = a3;
        v94 = (unsigned int *)v131;
        v95 = (unsigned int *)&v131[16 * v132];
        do
        {
          v96 = *((_QWORD *)v94 + 1);
          v97 = *v94;
          v94 += 4;
          sub_B99FD0((__int64)v55, v97, v96);
        }
        while ( v95 != v94 );
        a3 = v105;
      }
      goto LABEL_61;
    }
    if ( (unsigned __int8)sub_AC4810(0x27u) )
      v55 = (_BYTE *)sub_ADAB70(39, a4, v53, 0);
    else
      v55 = (_BYTE *)sub_AA93C0(0x27u, a4, (__int64)v53);
  }
  else
  {
    v55 = (_BYTE *)v54(v139, 39u, (_BYTE *)a4, (__int64)v53);
  }
  if ( !v55 )
    goto LABEL_104;
LABEL_61:
  v56 = sub_A81850((unsigned int **)&v131, v55, v108, (__int64)v121, 0, 0);
  v130 = 257;
  v25 = sub_B37A60((unsigned int **)&v131, v118 >> 2, v56, (__int64 *)&v127);
LABEL_27:
  nullsub_61();
  v141 = &unk_49DA100;
  nullsub_63();
  if ( v131 != &v133 )
    _libc_free((unsigned __int64)v131);
LABEL_31:
  v112 = *(__int64 ***)(v25 + 8);
  v26 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v26 + 8) - 17 <= 1 )
    v26 = **(_QWORD **)(v26 + 16);
  v27 = *(_DWORD *)(v26 + 8) >> 8;
  v28 = sub_BCE760(v112, v27);
  v107 = *(_QWORD *)(a3 + 8);
  if ( *(_BYTE *)a3 != 17 )
  {
    v102 = v28;
    v131 = "dst";
    v135 = 1;
    v32 = a1 + 24;
    v134 = 3;
    v33 = sub_BD2C40(72, 1u);
    v34 = v102;
    v35 = (__int64)v33;
    if ( v33 )
    {
      v103 = v33;
      sub_B51BF0((__int64)v33, a2, v34, (__int64)&v131, v32, 0);
      v35 = (__int64)v103;
    }
    sub_2CCA9F0(a1, (__int64)v112, v35, v109, v25, a6, (__int64)a7, a8);
    v135 = 1;
    v131 = "rem";
    v134 = 3;
    v36 = sub_AD64C0(v107, v118, 0);
    v113 = sub_B504D0(22, a3, v36, (__int64)&v131, v32, 0);
    v135 = 1;
    v131 = "offset";
    v134 = 3;
    v37 = sub_AD64C0(v107, v118, 0);
    v38 = sub_B504D0(17, v109, v37, (__int64)&v131, v32, 0);
    v127 = v129;
    v129[0] = v38;
    v128 = 0x100000001LL;
    v135 = 1;
    v131 = "rem.gep";
    v134 = 3;
    v119 = sub_BCB2B0(a7);
    v39 = sub_BD2C40(88, 2u);
    if ( !v39 )
    {
LABEL_43:
      sub_2CCA9F0(a1, v39[10], (__int64)v39, v113, a4, a6, (__int64)a7, a8);
      v41 = (unsigned __int64)v127;
      if ( v127 == v129 )
        return;
      goto LABEL_44;
    }
    v40 = *(_QWORD *)(a2 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v40 + 8) - 17 <= 1 )
    {
LABEL_42:
      sub_B44260((__int64)v39, v40, 34, 2u, v32, 0);
      v39[9] = v119;
      v39[10] = sub_B4DC50(v119, (__int64)v129, 1);
      sub_B4D9A0((__int64)v39, a2, v129, 1, (__int64)&v131);
      goto LABEL_43;
    }
    v42 = *(_QWORD *)(v129[0] + 8);
    v43 = *(unsigned __int8 *)(v42 + 8);
    if ( v43 == 17 )
    {
      v44 = 0;
    }
    else
    {
      if ( v43 != 18 )
        goto LABEL_42;
      v44 = 1;
    }
    v45 = *(_DWORD *)(v42 + 32);
    BYTE4(v124[0]) = v44;
    LODWORD(v124[0]) = v45;
    v40 = sub_BCE1B0((__int64 *)v40, v124[0]);
    goto LABEL_42;
  }
  v29 = *(unsigned __int64 **)(a3 + 24);
  if ( *(_DWORD *)(a3 + 32) <= 0x40u )
  {
    v30 = *(_QWORD *)(a3 + 24);
    if ( !v29 )
      return;
  }
  else
  {
    v30 = *v29;
    if ( !*v29 )
      return;
  }
  v101 = v28;
  v131 = "dst";
  v135 = 1;
  v134 = 3;
  v104 = a1 + 24;
  v57 = sub_BD2C40(72, 1u);
  v58 = (__int64)v57;
  if ( v57 )
    sub_B51BF0((__int64)v57, a2, v101, (__int64)&v131, v104, 0);
  sub_2CCA9F0(a1, (__int64)v112, v58, v109, v25, a6, (__int64)a7, a8);
  v100 = v30 % v118;
  if ( v100 )
  {
    v127 = v129;
    v128 = 0x100000000LL;
    v59 = *(_QWORD **)(v109 + 24);
    if ( *(_DWORD *)(v109 + 32) > 0x40u )
      v59 = (_QWORD *)*v59;
    v62 = sub_AD64C0(v107, v118 * (_QWORD)v59, 0);
    v63 = (unsigned int)v128;
    v64 = (unsigned int)v128 + 1LL;
    if ( v64 > HIDWORD(v128) )
    {
      sub_C8D5F0((__int64)&v127, v129, v64, 8u, v60, v61);
      v63 = (unsigned int)v128;
    }
    v127[v63] = v62;
    v131 = "rem.gep";
    v65 = v127;
    v98 = v128;
    LODWORD(v128) = v128 + 1;
    v135 = 1;
    v134 = 3;
    v114 = (unsigned int)v128;
    v110 = sub_BCB2B0(a7);
    v99 = v98 + 2;
    v66 = sub_BD2C40(88, v99);
    if ( v66 )
    {
      v67 = *(_QWORD *)(a2 + 8);
      v68 = v99 & 0x7FFFFFF;
      if ( (unsigned int)*(unsigned __int8 *)(v67 + 8) - 17 > 1 )
      {
        v87 = &v65[v114];
        if ( v65 != v87 )
        {
          v88 = v65;
          while ( 1 )
          {
            v89 = *(_QWORD *)(*v88 + 8);
            v90 = *(unsigned __int8 *)(v89 + 8);
            if ( v90 == 17 )
            {
              v91 = 0;
              goto LABEL_100;
            }
            if ( v90 == 18 )
              break;
            if ( v87 == ++v88 )
              goto LABEL_72;
          }
          v91 = 1;
LABEL_100:
          v92 = *(_DWORD *)(v89 + 32);
          BYTE4(v124[0]) = v91;
          LODWORD(v124[0]) = v92;
          v93 = sub_BCE1B0((__int64 *)v67, v124[0]);
          v68 = v99 & 0x7FFFFFF;
          v67 = v93;
        }
      }
LABEL_72:
      sub_B44260((__int64)v66, v67, 34, v68, v104, 0);
      v66[9] = v110;
      v66[10] = sub_B4DC50(v110, (__int64)v65, v114);
      sub_B4D9A0((__int64)v66, a2, v65, v114, (__int64)&v131);
    }
    if ( v12 )
    {
      v69 = (__int64 *)sub_BCB2B0(a7);
      v70 = sub_BCDA70(v69, v100);
      v71 = sub_BCE760((__int64 **)v70, v27);
      v135 = 1;
      v72 = v71;
      v134 = 3;
      v131 = "rem.dst";
      v73 = sub_BD2C40(72, 1u);
      v74 = (__int64)v73;
      if ( v73 )
        sub_B51BF0((__int64)v73, (__int64)v66, v72, (__int64)&v131, v104, 0);
      v75 = *(_BYTE *)(v70 + 8) == 12 ? sub_AD64C0(v70, 0, 0) : sub_AC9350((__int64 **)v70);
      v76 = -1;
      if ( v118 )
      {
        _BitScanReverse64(&v77, v118);
        v76 = 63 - (v77 ^ 0x3F);
      }
      v120 = v76;
      v78 = sub_BD2C40(80, unk_3F10A10);
      if ( v78 )
        sub_B4D3C0((__int64)v78, v75, v74, a6, v120, v79, v104, 0);
    }
    else
    {
      v85 = sub_AD64C0(v107, v100, 0);
      v86 = sub_BCB2B0(a7);
      sub_2CCA9F0(a1, v86, (__int64)v66, v85, a4, a6, (__int64)a7, a8);
    }
    v41 = (unsigned __int64)v127;
    if ( v127 != v129 )
LABEL_44:
      _libc_free(v41);
  }
}
