// Function: sub_385F910
// Address: 0x385f910
//
__int64 __fastcall sub_385F910(
        __int64 a1,
        __int64 *a2,
        __m128i a3,
        __m128i a4,
        __int64 a5,
        __int64 *a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v8; // rbx
  __int64 v9; // r12
  unsigned __int64 v10; // r11
  unsigned __int64 v11; // r15
  __int64 v12; // rax
  unsigned int v15; // edx
  __int64 v16; // rax
  unsigned int v17; // edx
  __int64 result; // rax
  char v19; // bl
  __int64 v20; // r12
  __int64 v21; // r14
  __int64 *v22; // rax
  __int64 v23; // rdx
  __int64 *v24; // r11
  __int64 v25; // rsi
  __int64 v26; // r9
  __int64 v27; // rax
  __int64 v28; // r14
  __int64 v29; // rax
  __int64 v30; // r15
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // r11
  unsigned __int64 v33; // r10
  __int64 v34; // rax
  unsigned int v35; // esi
  unsigned __int64 v36; // r9
  const void **v37; // rdi
  unsigned __int64 v38; // r8
  unsigned __int64 v39; // rcx
  unsigned int v40; // ecx
  __int64 v41; // rax
  unsigned __int64 v42; // rdx
  int v43; // edx
  unsigned __int64 v44; // r10
  unsigned __int64 v45; // r14
  int v46; // ecx
  int v47; // eax
  __int64 v48; // rcx
  unsigned __int64 v49; // rcx
  unsigned __int64 v50; // rdx
  unsigned __int64 v51; // rax
  unsigned __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // r9
  __int64 v58; // r14
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rax
  int v62; // eax
  unsigned __int64 v63; // rbx
  __int64 v64; // r14
  _QWORD *v65; // r12
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // r9
  unsigned int v69; // eax
  __int64 v70; // r9
  __int64 v71; // rsi
  unsigned __int64 v72; // rbx
  __int64 v73; // rax
  unsigned __int64 v74; // rax
  __int64 v75; // rbx
  unsigned int v76; // eax
  __int64 v77; // r9
  __int64 v78; // rsi
  unsigned __int64 v79; // rcx
  int v80; // eax
  __int64 v81; // rax
  int v82; // eax
  unsigned __int64 v83; // rax
  _QWORD *v84; // rax
  __int64 v85; // rax
  unsigned int v86; // esi
  int v87; // eax
  unsigned __int64 v88; // rax
  _QWORD *v89; // rax
  __int64 *v90; // r12
  __int64 v91; // rax
  unsigned __int64 *v92; // [rsp+8h] [rbp-88h]
  unsigned __int64 v93; // [rsp+10h] [rbp-80h]
  unsigned __int64 v94; // [rsp+20h] [rbp-70h]
  unsigned __int64 v95; // [rsp+20h] [rbp-70h]
  unsigned __int64 v96; // [rsp+20h] [rbp-70h]
  char v97; // [rsp+28h] [rbp-68h]
  unsigned __int64 v98; // [rsp+28h] [rbp-68h]
  __int64 v99; // [rsp+28h] [rbp-68h]
  __int64 v100; // [rsp+28h] [rbp-68h]
  unsigned __int64 v101; // [rsp+28h] [rbp-68h]
  unsigned __int64 v102; // [rsp+28h] [rbp-68h]
  unsigned __int64 v103; // [rsp+30h] [rbp-60h]
  __int64 v104; // [rsp+30h] [rbp-60h]
  __int64 v105; // [rsp+30h] [rbp-60h]
  __int64 v106; // [rsp+30h] [rbp-60h]
  __int64 v107; // [rsp+30h] [rbp-60h]
  __int64 v108; // [rsp+30h] [rbp-60h]
  __int64 v109; // [rsp+30h] [rbp-60h]
  __int64 v110; // [rsp+30h] [rbp-60h]
  unsigned __int64 v111; // [rsp+38h] [rbp-58h]
  __int64 *v112; // [rsp+38h] [rbp-58h]
  __int64 *v113; // [rsp+38h] [rbp-58h]
  __int64 v114; // [rsp+38h] [rbp-58h]
  unsigned __int64 v115; // [rsp+38h] [rbp-58h]
  __int64 v116; // [rsp+38h] [rbp-58h]
  __int64 v117; // [rsp+38h] [rbp-58h]
  unsigned __int64 v118; // [rsp+38h] [rbp-58h]
  __int64 v119; // [rsp+38h] [rbp-58h]
  unsigned __int64 v120; // [rsp+38h] [rbp-58h]
  __int64 v121; // [rsp+38h] [rbp-58h]
  __int64 v122; // [rsp+38h] [rbp-58h]
  unsigned __int64 v123; // [rsp+38h] [rbp-58h]
  unsigned __int64 v124; // [rsp+40h] [rbp-50h] BYREF
  __int64 v125; // [rsp+48h] [rbp-48h]
  _QWORD v126[8]; // [rsp+50h] [rbp-40h] BYREF

  v8 = *a2 >> 2;
  v9 = *a6 >> 2;
  v10 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  v11 = *a6 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (((unsigned __int8)v9 | (unsigned __int8)v8) & 1) == 0 )
    return 0;
  v12 = *(_QWORD *)v10;
  if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) == 16 )
    v12 = **(_QWORD **)(v12 + 16);
  v15 = *(_DWORD *)(v12 + 8);
  v16 = *(_QWORD *)v11;
  v17 = v15 >> 8;
  if ( *(_BYTE *)(*(_QWORD *)v11 + 8LL) == 16 )
    v16 = **(_QWORD **)(v16 + 16);
  if ( v17 != *(_DWORD *)(v16 + 8) >> 8 )
    return 1;
  v111 = *a2 & 0xFFFFFFFFFFFFFFF8LL;
  v97 = v9 & 1;
  v19 = v8 & 1;
  v20 = sub_385E580(*(_QWORD *)a1, v111, *(_QWORD *)(a1 + 8), a8, 1, 1, a3, a4);
  v21 = sub_385E580(*(_QWORD *)a1, v11, *(_QWORD *)(a1 + 8), a8, 1, 1, a3, a4);
  v103 = v111;
  v112 = sub_1494E70(*(_QWORD *)a1, v111, a3, a4);
  v22 = sub_1494E70(*(_QWORD *)a1, v11, a3, a4);
  v23 = (__int64)v112;
  v24 = (__int64 *)v103;
  v25 = (__int64)v22;
  if ( v20 < 0 )
  {
    v23 = (__int64)v22;
    v25 = (__int64)v112;
    v53 = v20;
    v20 = v21;
    v21 = v53;
    LOBYTE(v53) = v19;
    v19 = v97;
    v97 = v53;
    v24 = (__int64 *)v11;
    v11 = v103;
  }
  v113 = v24;
  v26 = sub_14806B0(*(_QWORD *)(*(_QWORD *)a1 + 112LL), v25, v23, 0, 0);
  if ( v20 == 0 || v21 == 0 || v20 != v21 )
    return 1;
  v27 = *v113;
  v114 = v26;
  v28 = **(_QWORD **)(v27 + 16);
  v104 = **(_QWORD **)(*(_QWORD *)v11 + 16LL);
  v29 = sub_157EB90(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 32LL));
  v30 = sub_1632FA0(v29);
  v31 = sub_12BE0A0(v30, v28);
  v32 = v31;
  v33 = abs64(v20);
  if ( *(_WORD *)(v114 + 24) )
  {
    v95 = v33;
    v98 = v31;
    v63 = sub_12BE0A0(v30, v104);
    if ( v63 == v98 )
    {
      v64 = sub_1495DC0(*(_QWORD *)a1, a3, a4);
      v65 = *(_QWORD **)(*(_QWORD *)a1 + 112LL);
      v66 = sub_1456040(v64);
      v126[1] = sub_145CF80((__int64)v65, v66, v95 * v63, 0);
      v126[0] = v64;
      v124 = (unsigned __int64)v126;
      v125 = 0x200000002LL;
      v67 = sub_147EE30(v65, (__int64 **)&v124, 0, 0, a3, a4);
      v68 = v114;
      v58 = v67;
      if ( (_QWORD *)v124 != v126 )
      {
        _libc_free(v124);
        v68 = v114;
      }
      v99 = v68;
      v105 = sub_1456040(v68);
      v69 = sub_15A9FE0(v30, v105);
      v70 = v99;
      v71 = v105;
      v119 = 1;
      v72 = v69;
      while ( 2 )
      {
        switch ( *(_BYTE *)(v71 + 8) )
        {
          case 0:
          case 8:
          case 0xA:
          case 0xC:
          case 0x10:
            v81 = v119 * *(_QWORD *)(v71 + 32);
            v71 = *(_QWORD *)(v71 + 24);
            v119 = v81;
            continue;
          case 1:
            v73 = 16;
            goto LABEL_69;
          case 2:
            v73 = 32;
            goto LABEL_69;
          case 3:
          case 9:
            v73 = 64;
            goto LABEL_69;
          case 4:
            v73 = 80;
            goto LABEL_69;
          case 5:
          case 6:
            v73 = 128;
            goto LABEL_69;
          case 7:
            v80 = sub_15A9520(v30, 0);
            v70 = v99;
            v73 = (unsigned int)(8 * v80);
            goto LABEL_69;
          case 0xB:
            v73 = *(_DWORD *)(v71 + 8) >> 8;
            goto LABEL_69;
          case 0xD:
            v84 = (_QWORD *)sub_15A9930(v30, v71);
            v70 = v99;
            v73 = 8LL * *v84;
            goto LABEL_69;
          case 0xE:
            v107 = *(_QWORD *)(v71 + 32);
            v83 = sub_12BE0A0(v30, *(_QWORD *)(v71 + 24));
            v70 = v99;
            v73 = 8 * v107 * v83;
            goto LABEL_69;
          case 0xF:
            v82 = sub_15A9520(v30, *(_DWORD *)(v71 + 8) >> 8);
            v70 = v99;
            v73 = (unsigned int)(8 * v82);
LABEL_69:
            v100 = v70;
            v74 = v72 * ((v72 + ((unsigned __int64)(v119 * v73 + 7) >> 3) - 1) / v72);
            v75 = 1;
            v120 = v74;
            v106 = sub_1456040(v58);
            v76 = sub_15A9FE0(v30, v106);
            v77 = v100;
            v78 = v106;
            v79 = v76;
            while ( 2 )
            {
              switch ( *(_BYTE *)(v78 + 8) )
              {
                case 1:
                  v54 = 16;
                  goto LABEL_43;
                case 2:
                  v54 = 32;
                  goto LABEL_43;
                case 3:
                case 9:
                  v54 = 64;
                  goto LABEL_43;
                case 4:
                  v54 = 80;
                  goto LABEL_43;
                case 5:
                case 6:
                  v54 = 128;
                  goto LABEL_43;
                case 7:
                  v101 = v79;
                  v86 = 0;
                  v108 = v77;
                  goto LABEL_82;
                case 0xB:
                  v54 = *(_DWORD *)(v78 + 8) >> 8;
                  goto LABEL_43;
                case 0xD:
                  v102 = v79;
                  v110 = v77;
                  v89 = (_QWORD *)sub_15A9930(v30, v78);
                  v77 = v110;
                  v79 = v102;
                  v54 = 8LL * *v89;
                  goto LABEL_43;
                case 0xE:
                  v96 = v79;
                  v109 = *(_QWORD *)(v78 + 32);
                  v88 = sub_12BE0A0(v30, *(_QWORD *)(v78 + 24));
                  v77 = v100;
                  v79 = v96;
                  v54 = 8 * v109 * v88;
                  goto LABEL_43;
                case 0xF:
                  v101 = v79;
                  v108 = v77;
                  v86 = *(_DWORD *)(v78 + 8) >> 8;
LABEL_82:
                  v87 = sub_15A9520(v30, v86);
                  v77 = v108;
                  v79 = v101;
                  v54 = (unsigned int)(8 * v87);
LABEL_43:
                  if ( v79 * ((v79 + ((unsigned __int64)(v75 * v54 + 7) >> 3) - 1) / v79) >= v120 )
                  {
                    v122 = v77;
                    v91 = sub_1456040(v58);
                    v57 = sub_147BE00((__int64)v65, v122, v91);
                  }
                  else
                  {
                    v116 = v77;
                    v55 = sub_1456040(v77);
                    v56 = sub_14747F0((__int64)v65, v58, v55, 0);
                    v57 = v116;
                    v58 = v56;
                  }
                  v117 = v57;
                  v59 = sub_14806B0((__int64)v65, v57, v58, 0, 0);
                  if ( (unsigned __int8)sub_1477C30((__int64)v65, v59) )
                    return 0;
                  v60 = sub_1480620((__int64)v65, v117, 0);
                  v61 = sub_14806B0((__int64)v65, v60, v58, 0, 0);
                  if ( (unsigned __int8)sub_1477C30((__int64)v65, v61) )
                    return 0;
                  goto LABEL_61;
                case 0x10:
                  v85 = *(_QWORD *)(v78 + 32);
                  v78 = *(_QWORD *)(v78 + 24);
                  v75 *= v85;
                  continue;
                default:
                  sub_1495DC0(v30, a3, a4);
                  BUG();
              }
            }
        }
      }
    }
LABEL_61:
    *(_BYTE *)(a1 + 216) = 1;
    return 1;
  }
  v34 = *(_QWORD *)(v114 + 32);
  v35 = *(_DWORD *)(v34 + 32);
  v36 = *(_QWORD *)(v34 + 24);
  v37 = (const void **)(v34 + 24);
  if ( v35 > 0x40 )
  {
    v38 = *(_QWORD *)v36;
    if ( !*(_QWORD *)v36 || (v39 = abs64(v38), v33 <= 1) || v104 != v28 || v39 % v32 )
    {
      v40 = v35 - 1;
      v41 = 1LL << ((unsigned __int8)v35 - 1);
      goto LABEL_53;
    }
  }
  else
  {
    v38 = (__int64)(v36 << (64 - (unsigned __int8)v35)) >> (64 - (unsigned __int8)v35);
    if ( !v38 || (v39 = abs64(v38), v33 <= 1) || v104 != v28 || v39 % v32 )
    {
      v41 = 1LL << ((unsigned __int8)v35 - 1);
LABEL_21:
      v42 = v36;
      if ( (v41 & v36) == 0 )
      {
LABEL_22:
        if ( v42 )
          goto LABEL_23;
        if ( v104 != v28 )
          return 1;
        return 2;
      }
LABEL_56:
      if ( !v19 || v97 || !byte_5052040 )
        return 2;
      if ( v35 > 0x40 )
      {
        LODWORD(v125) = v35;
        v123 = v32;
        sub_16A4FD0((__int64)&v124, v37);
        LOBYTE(v35) = v125;
        v32 = v123;
        if ( (unsigned int)v125 > 0x40 )
        {
          sub_16A8F40((__int64 *)&v124);
          v32 = v123;
LABEL_96:
          v121 = v32;
          sub_16A7400((__int64)&v124);
          v90 = (__int64 *)v124;
          if ( (unsigned int)v125 <= 0x40 )
          {
            if ( !sub_385F870(a1, v124, v121) && v104 == v28 )
              return 2;
          }
          else
          {
            if ( !sub_385F870(a1, *(_QWORD *)v124, v121) && v104 == v28 )
            {
              if ( v90 )
                j_j___libc_free_0_0((unsigned __int64)v90);
              return 2;
            }
            if ( v90 )
            {
              j_j___libc_free_0_0((unsigned __int64)v90);
              return 3;
            }
          }
          return 3;
        }
      }
      else
      {
        LODWORD(v125) = v35;
        v124 = v36;
      }
      v124 = ~v124 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v35);
      goto LABEL_96;
    }
  }
  if ( v39 / v32 % v33 )
    return 0;
  v40 = v35 - 1;
  v41 = 1LL << ((unsigned __int8)v35 - 1);
  if ( v35 <= 0x40 )
    goto LABEL_21;
LABEL_53:
  if ( (*(_QWORD *)(v36 + 8LL * (v40 >> 6)) & v41) != 0 )
    goto LABEL_56;
  v93 = v38;
  v94 = v33;
  v118 = v32;
  v92 = (unsigned __int64 *)v36;
  v62 = sub_16A57B0((__int64)v37);
  v32 = v118;
  v33 = v94;
  v38 = v93;
  if ( v35 - v62 <= 0x40 )
  {
    v42 = *v92;
    goto LABEL_22;
  }
LABEL_23:
  if ( v104 != v28 )
    return 1;
  v43 = 1;
  v44 = v32 * v33;
  v45 = v44;
  v46 = 1;
  if ( dword_50523E8[0] )
    v46 = dword_50523E8[0];
  if ( dword_50524C8[0] )
    v43 = dword_50524C8[0];
  v47 = v43 * v46;
  if ( (unsigned int)(v43 * v46) < 2 )
    v47 = 2;
  v48 = (unsigned int)(v47 - 1);
  result = 4;
  v49 = v32 + v44 * v48;
  if ( v38 >= v49 )
  {
    v50 = *(_QWORD *)(a1 + 200);
    if ( v50 >= v49 )
    {
      v51 = *(_QWORD *)(a1 + 200);
      if ( v38 <= v50 )
        v51 = v38;
      *(_QWORD *)(a1 + 200) = v51;
      if ( !v19 && v97 && byte_5052040 )
      {
        v115 = v32;
        if ( sub_385F870(a1, v38, v32) )
          return 6;
        v51 = *(_QWORD *)(a1 + 200);
        v32 = v115;
      }
      v52 = 8 * v32 * (v51 / v45);
      if ( v52 > *(_QWORD *)(a1 + 208) )
        v52 = *(_QWORD *)(a1 + 208);
      *(_QWORD *)(a1 + 208) = v52;
      return 5;
    }
  }
  return result;
}
