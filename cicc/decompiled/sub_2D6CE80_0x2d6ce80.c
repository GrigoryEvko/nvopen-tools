// Function: sub_2D6CE80
// Address: 0x2d6ce80
//
__int64 __fastcall sub_2D6CE80(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v5; // rdx
  __int64 v6; // rbx
  unsigned __int16 v7; // ax
  __int64 v8; // r15
  __int64 v9; // r13
  unsigned __int8 *v10; // r12
  __int64 v11; // rbx
  char v12; // al
  __int64 v13; // rax
  int v14; // r10d
  _QWORD *v15; // r12
  unsigned int v16; // ecx
  _QWORD *v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 *v20; // r12
  __int64 v21; // rdx
  __int64 v22; // rdx
  unsigned __int16 v24; // ax
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // r14
  unsigned __int8 *v28; // r15
  __int64 v29; // r12
  unsigned int v30; // eax
  __int64 v31; // r13
  unsigned __int16 v32; // ax
  unsigned __int64 v33; // rax
  int v34; // esi
  int v35; // r13d
  unsigned __int64 *v36; // rdi
  __int64 v37; // rcx
  _QWORD *v38; // rdx
  unsigned __int64 v39; // r8
  __int64 *v40; // r13
  int v41; // esi
  int v42; // r15d
  unsigned __int64 *v43; // rdi
  unsigned __int64 v44; // rcx
  unsigned int v45; // edx
  _QWORD *v46; // rax
  unsigned __int64 v47; // r8
  unsigned int v48; // edx
  __int64 v49; // r8
  bool v50; // r13
  int v51; // ecx
  __int16 v52; // dx
  unsigned __int64 *v53; // r14
  __int64 v54; // rax
  char v55; // dl
  __int64 v56; // rsi
  bool v57; // zf
  __int64 v58; // rax
  __int64 v59; // r14
  __int64 v60; // rsi
  __int64 v61; // rdx
  unsigned int v62; // r14d
  __int64 v63; // rsi
  unsigned __int8 *v64; // rsi
  unsigned int v65; // edx
  unsigned __int64 v66; // rdi
  unsigned __int8 v67; // dl
  char v68; // dh
  unsigned __int64 *v69; // r15
  char v70; // al
  __int64 v71; // rcx
  __int64 v72; // rsi
  __int64 v73; // rax
  __int64 v74; // rcx
  __int64 v75; // rsi
  __int64 *v76; // r8
  __int64 v77; // rax
  unsigned __int64 *v78; // r15
  _QWORD *v79; // rax
  __int64 v80; // r13
  __int64 v81; // rdx
  int v82; // eax
  int v83; // edx
  int v84; // r9d
  __int64 v85; // r8
  __int64 v86; // rsi
  unsigned __int8 *v87; // rsi
  __int64 v88; // rsi
  unsigned __int8 *v89; // rsi
  __int64 *v90; // [rsp+0h] [rbp-140h]
  __int64 v91; // [rsp+8h] [rbp-138h]
  __int64 v92; // [rsp+8h] [rbp-138h]
  __int64 *v93; // [rsp+10h] [rbp-130h]
  unsigned __int8 *v94; // [rsp+18h] [rbp-128h]
  __int64 v95; // [rsp+20h] [rbp-120h]
  __int64 *v96; // [rsp+28h] [rbp-118h]
  __int64 v97; // [rsp+30h] [rbp-110h]
  unsigned __int8 v98; // [rsp+38h] [rbp-108h]
  const void **v99; // [rsp+38h] [rbp-108h]
  unsigned __int64 v100; // [rsp+38h] [rbp-108h]
  bool v104; // [rsp+67h] [rbp-D9h]
  unsigned __int8 v105; // [rsp+68h] [rbp-D8h]
  __int64 v106; // [rsp+68h] [rbp-D8h]
  __int64 v107; // [rsp+68h] [rbp-D8h]
  __int64 v108; // [rsp+68h] [rbp-D8h]
  __int64 v109; // [rsp+68h] [rbp-D8h]
  __int64 v111; // [rsp+78h] [rbp-C8h]
  __int64 v112; // [rsp+88h] [rbp-B8h] BYREF
  unsigned __int64 v113; // [rsp+90h] [rbp-B0h] BYREF
  unsigned int v114; // [rsp+98h] [rbp-A8h]
  __int64 v115; // [rsp+A0h] [rbp-A0h] BYREF
  __int64 v116; // [rsp+A8h] [rbp-98h]
  __int64 v117; // [rsp+B0h] [rbp-90h]
  unsigned int v118; // [rsp+B8h] [rbp-88h]
  __int64 v119; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v120; // [rsp+C8h] [rbp-78h]
  __int64 v121; // [rsp+D0h] [rbp-70h]
  unsigned int v122; // [rsp+D8h] [rbp-68h]
  unsigned __int64 *v123; // [rsp+E0h] [rbp-60h] BYREF
  unsigned int v124; // [rsp+E8h] [rbp-58h]
  __int16 v125; // [rsp+100h] [rbp-40h]

  v5 = *(__int64 **)(a1 + 8);
  v111 = *(_QWORD *)(a1 + 40);
  v6 = a3;
  v115 = 0;
  v116 = 0;
  v117 = 0;
  v118 = 0;
  v7 = sub_2D5BAE0(a3, a4, v5, 0);
  v104 = 0;
  if ( v7 )
    v104 = *(_QWORD *)(v6 + 8LL * v7 + 112) != 0;
  v105 = 0;
  v8 = *(_QWORD *)(a1 + 16);
  if ( !v8 )
  {
LABEL_97:
    sub_F54ED0((unsigned __int8 *)a1);
    sub_B43D60((_QWORD *)a1);
    v105 = 1;
    goto LABEL_23;
  }
LABEL_9:
  while ( 2 )
  {
    v10 = *(unsigned __int8 **)(v8 + 24);
    v11 = v8;
    v8 = *(_QWORD *)(v8 + 8);
    v12 = *v10;
    if ( *v10 == 84 )
      goto LABEL_8;
    if ( v12 != 67 )
    {
      if ( v12 != 57 )
        goto LABEL_8;
      if ( (v10[7] & 0x40) != 0 )
      {
        v9 = *(_QWORD *)(*((_QWORD *)v10 - 1) + 32LL);
        if ( *(_BYTE *)v9 != 17 )
          goto LABEL_8;
      }
      else
      {
        v9 = *(_QWORD *)&v10[-32 * (*((_DWORD *)v10 + 1) & 0x7FFFFFF) + 32];
        if ( *(_BYTE *)v9 != 17 )
          goto LABEL_8;
      }
      v99 = (const void **)(v9 + 24);
      v114 = *(_DWORD *)(v9 + 32);
      if ( v114 > 0x40 )
        sub_C43780((__int64)&v113, v99);
      else
        v113 = *(_QWORD *)(v9 + 24);
      sub_C46A40((__int64)&v113, 1);
      v48 = v114;
      v114 = 0;
      LODWORD(v120) = v48;
      v119 = v113;
      if ( v48 > 0x40 )
      {
        sub_C43B90(&v119, (__int64 *)v99);
        v62 = v120;
        v49 = v119;
        LODWORD(v120) = 0;
        v124 = v62;
        v123 = (unsigned __int64 *)v119;
        if ( v62 > 0x40 )
        {
          v100 = v119;
          v50 = v62 == (unsigned int)sub_C444A0((__int64)&v123);
          if ( v100 )
          {
            j_j___libc_free_0_0(v100);
            if ( (unsigned int)v120 > 0x40 )
            {
              if ( v119 )
                j_j___libc_free_0_0(v119);
            }
          }
LABEL_54:
          if ( v114 > 0x40 && v113 )
            j_j___libc_free_0_0(v113);
          if ( !v50 )
            goto LABEL_8;
          goto LABEL_11;
        }
      }
      else
      {
        v49 = *(_QWORD *)(v9 + 24) & v113;
      }
      v50 = v49 == 0;
      goto LABEL_54;
    }
LABEL_11:
    v13 = *((_QWORD *)v10 + 5);
    v112 = v13;
    if ( v111 == v13 )
    {
      v98 = v104 && *v10 == 67;
      if ( !v98 )
        goto LABEL_8;
      v24 = sub_2D5BAE0(a3, a4, *((__int64 **)v10 + 1), 0);
      if ( v24 )
      {
        if ( *(_QWORD *)(a3 + 8LL * v24 + 112) )
          goto LABEL_8;
      }
      v25 = *((_QWORD *)v10 + 5);
      v119 = 0;
      v26 = 0;
      v120 = 0;
      v121 = 0;
      v122 = 0;
      v27 = *((_QWORD *)v10 + 2);
      v97 = v25;
      v105 = 0;
      if ( !v27 )
        goto LABEL_46;
      v94 = v10;
      v95 = v8;
      while ( 1 )
      {
        v28 = *(unsigned __int8 **)(v27 + 24);
        v29 = v27;
        v27 = *(_QWORD *)(v27 + 8);
        v30 = sub_2FEBEF0(a3, (unsigned int)*v28 - 29);
        v31 = v30;
        if ( !v30 )
          goto LABEL_44;
        v32 = sub_2D5BAE0(a3, a4, *((__int64 **)v28 + 1), 1);
        if ( (v32 == 1 || v32 && *(_QWORD *)(a3 + 8LL * v32 + 112))
          && ((unsigned int)v31 > 0x1F3 || (*(_BYTE *)(v31 + a3 + 500LL * v32 + 6414) & 0xFB) == 0) )
        {
          goto LABEL_44;
        }
        if ( *v28 == 84 )
          goto LABEL_44;
        v33 = *((_QWORD *)v28 + 5);
        v113 = v33;
        if ( v97 == v33 )
          goto LABEL_44;
        v34 = v118;
        if ( !v118 )
          break;
        v35 = 1;
        v36 = 0;
        LODWORD(v37) = (v118 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
        v38 = (_QWORD *)(v116 + 16LL * (unsigned int)v37);
        v39 = *v38;
        if ( v33 == *v38 )
        {
LABEL_39:
          v40 = v38 + 1;
          goto LABEL_40;
        }
        while ( v39 != -4096 )
        {
          if ( !v36 && v39 == -8192 )
            v36 = v38;
          v37 = (v118 - 1) & ((_DWORD)v37 + v35);
          v38 = (_QWORD *)(v116 + 16 * v37);
          v39 = *v38;
          if ( v33 == *v38 )
            goto LABEL_39;
          ++v35;
        }
        if ( !v36 )
          v36 = v38;
        ++v115;
        v83 = v117 + 1;
        v123 = v36;
        if ( 4 * ((int)v117 + 1) >= 3 * v118 )
          goto LABEL_150;
        if ( v118 - HIDWORD(v117) - v83 <= v118 >> 3 )
          goto LABEL_151;
LABEL_146:
        LODWORD(v117) = v83;
        if ( *v36 != -4096 )
          --HIDWORD(v117);
        *v36 = v33;
        v40 = (__int64 *)(v36 + 1);
        v36[1] = 0;
LABEL_40:
        v41 = v122;
        if ( !v122 )
        {
          ++v119;
          v123 = 0;
LABEL_134:
          v41 = 2 * v122;
LABEL_135:
          sub_2D6BA80((__int64)&v119, v41);
          sub_2D64D20((__int64)&v119, (__int64 *)&v113, &v123);
          v44 = v113;
          v43 = v123;
          v82 = v121 + 1;
          goto LABEL_130;
        }
        v42 = 1;
        v43 = 0;
        v44 = v113;
        v45 = (v122 - 1) & (((unsigned int)v113 >> 9) ^ ((unsigned int)v113 >> 4));
        v46 = (_QWORD *)(v120 + 16LL * v45);
        v47 = *v46;
        if ( v113 == *v46 )
        {
LABEL_42:
          v96 = v46 + 1;
          goto LABEL_43;
        }
        while ( v47 != -4096 )
        {
          if ( v47 == -8192 && !v43 )
            v43 = v46;
          v45 = (v122 - 1) & (v42 + v45);
          v46 = (_QWORD *)(v120 + 16LL * v45);
          v47 = *v46;
          if ( v113 == *v46 )
            goto LABEL_42;
          ++v42;
        }
        if ( !v43 )
          v43 = v46;
        ++v119;
        v82 = v121 + 1;
        v123 = v43;
        if ( 4 * ((int)v121 + 1) >= 3 * v122 )
          goto LABEL_134;
        if ( v122 - HIDWORD(v121) - v82 <= v122 >> 3 )
          goto LABEL_135;
LABEL_130:
        LODWORD(v121) = v82;
        if ( *v43 != -4096 )
          --HIDWORD(v121);
        *v43 = v44;
        v43[1] = 0;
        v96 = (__int64 *)(v43 + 1);
LABEL_43:
        if ( *v40 || *v96 )
          goto LABEL_44;
        v69 = (unsigned __int64 *)sub_AA5190(v113);
        if ( v69 )
        {
          v70 = v68;
        }
        else
        {
          v70 = 0;
          v67 = 0;
        }
        v71 = v67;
        BYTE1(v71) = v70;
        v108 = v71;
        v57 = *(_BYTE *)a1 == 56;
        v72 = *(_QWORD *)(a1 - 64);
        v125 = 257;
        if ( v57 )
          v73 = sub_B504D0(27, v72, a2, (__int64)&v123, 0, 0);
        else
          v73 = sub_B504D0(26, v72, a2, (__int64)&v123, 0, 0);
        *v40 = v73;
        v74 = *v40;
        v75 = *(_QWORD *)(a1 + 48);
        v76 = (__int64 *)(*v40 + 48);
        v123 = (unsigned __int64 *)v75;
        if ( !v75 )
        {
          if ( v76 == (__int64 *)&v123 )
            goto LABEL_114;
          v88 = *(_QWORD *)(v74 + 48);
          if ( !v88 )
            goto LABEL_114;
LABEL_164:
          v92 = v74;
          v93 = v76;
          sub_B91220((__int64)v76, v88);
          v74 = v92;
          v76 = v93;
          goto LABEL_165;
        }
        v90 = v76;
        v91 = v74;
        sub_B96E90((__int64)&v123, v75, 1);
        v76 = v90;
        if ( v90 == (__int64 *)&v123 )
        {
          if ( v123 )
            sub_B91220((__int64)&v123, (__int64)v123);
          goto LABEL_114;
        }
        v74 = v91;
        v88 = *(_QWORD *)(v91 + 48);
        if ( v88 )
          goto LABEL_164;
LABEL_165:
        v89 = (unsigned __int8 *)v123;
        *(_QWORD *)(v74 + 48) = v123;
        if ( v89 )
          sub_B976B0((__int64)&v123, v89, (__int64)v76);
LABEL_114:
        sub_B44150((_QWORD *)*v40, v113, v69, v108);
        v77 = sub_AA5190(v113);
        if ( !v77 )
          BUG();
        v78 = *(unsigned __int64 **)(v77 + 8);
        v125 = 257;
        v79 = (_QWORD *)sub_B51D30((unsigned int)*v94 - 29, *v40, *((_QWORD *)v94 + 1), (__int64)&v123, 0, 0);
        *v96 = (__int64)v79;
        sub_B44150(v79, v113, v78, 1);
        v80 = *v96;
        v123 = (unsigned __int64 *)*((_QWORD *)v94 + 6);
        if ( v123 )
        {
          sub_B96E90((__int64)&v123, (__int64)v123, 1);
          v81 = v80 + 48;
          if ( (unsigned __int64 **)(v80 + 48) != &v123 )
          {
            v86 = *(_QWORD *)(v80 + 48);
            if ( v86 )
            {
LABEL_159:
              v109 = v81;
              sub_B91220(v81, v86);
              v81 = v109;
            }
            v87 = (unsigned __int8 *)v123;
            *(_QWORD *)(v80 + 48) = v123;
            if ( v87 )
              sub_B976B0((__int64)&v123, v87, v81);
            goto LABEL_119;
          }
          if ( v123 )
            sub_B91220((__int64)&v123, (__int64)v123);
        }
        else
        {
          v81 = v80 + 48;
          if ( (unsigned __int64 **)(v80 + 48) != &v123 )
          {
            v86 = *(_QWORD *)(v80 + 48);
            if ( v86 )
              goto LABEL_159;
          }
        }
LABEL_119:
        sub_AC2B30(v29, *v96);
        v105 = v98;
LABEL_44:
        if ( !v27 )
        {
          v8 = v95;
          v27 = v120;
          v26 = 16LL * v122;
LABEL_46:
          sub_C7D6A0(v27, v26, 8);
LABEL_8:
          if ( !v8 )
            goto LABEL_22;
          goto LABEL_9;
        }
      }
      ++v115;
      v123 = 0;
LABEL_150:
      v34 = 2 * v118;
LABEL_151:
      sub_2D6CCA0((__int64)&v115, v34);
      sub_2D65270((__int64)&v115, (__int64 *)&v113, &v123);
      v33 = v113;
      v36 = v123;
      v83 = v117 + 1;
      goto LABEL_146;
    }
    if ( !v118 )
    {
      ++v115;
      v123 = 0;
      goto LABEL_101;
    }
    v14 = 1;
    v15 = 0;
    v16 = (v118 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
    v17 = (_QWORD *)(v116 + 16LL * v16);
    v18 = *v17;
    if ( v13 != *v17 )
    {
      while ( v18 != -4096 )
      {
        if ( v18 == -8192 && !v15 )
          v15 = v17;
        v16 = (v118 - 1) & (v14 + v16);
        v17 = (_QWORD *)(v116 + 16LL * v16);
        v18 = *v17;
        if ( v13 == *v17 )
          goto LABEL_14;
        ++v14;
      }
      if ( !v15 )
        v15 = v17;
      ++v115;
      v51 = v117 + 1;
      v123 = v15;
      if ( 4 * ((int)v117 + 1) < 3 * v118 )
      {
        if ( v118 - HIDWORD(v117) - v51 <= v118 >> 3 )
        {
          sub_2D6CCA0((__int64)&v115, v118);
          sub_2D65270((__int64)&v115, &v112, &v123);
          v13 = v112;
          v15 = v123;
          v51 = v117 + 1;
        }
LABEL_70:
        LODWORD(v117) = v51;
        if ( *v15 != -4096 )
          --HIDWORD(v117);
        *v15 = v13;
        v20 = v15 + 1;
        *v20 = 0;
        goto LABEL_73;
      }
LABEL_101:
      sub_2D6CCA0((__int64)&v115, 2 * v118);
      if ( v118 )
      {
        v13 = v112;
        v65 = (v118 - 1) & (((unsigned int)v112 >> 9) ^ ((unsigned int)v112 >> 4));
        v15 = (_QWORD *)(v116 + 16LL * v65);
        v66 = *v15;
        if ( v112 == *v15 )
        {
LABEL_103:
          v123 = v15;
          v51 = v117 + 1;
        }
        else
        {
          v84 = 1;
          v85 = 0;
          while ( v66 != -4096 )
          {
            if ( !v85 && v66 == -8192 )
              v85 = (__int64)v15;
            v65 = (v118 - 1) & (v84 + v65);
            v15 = (_QWORD *)(v116 + 16LL * v65);
            v66 = *v15;
            if ( v112 == *v15 )
              goto LABEL_103;
            ++v84;
          }
          if ( !v85 )
            v85 = (__int64)v15;
          v51 = v117 + 1;
          v123 = (unsigned __int64 *)v85;
          v15 = (_QWORD *)v85;
        }
      }
      else
      {
        v123 = 0;
        v15 = 0;
        v13 = v112;
        v51 = v117 + 1;
      }
      goto LABEL_70;
    }
LABEL_14:
    v19 = v17[1];
    v20 = v17 + 1;
    if ( v19 )
    {
      if ( *(_QWORD *)v11 )
      {
        v21 = *(_QWORD *)(v11 + 8);
        **(_QWORD **)(v11 + 16) = v21;
        if ( v21 )
          goto LABEL_17;
      }
      *(_QWORD *)v11 = v19;
      goto LABEL_19;
    }
LABEL_73:
    v53 = (unsigned __int64 *)sub_AA5190(v112);
    if ( v53 )
    {
      LOBYTE(v54) = v52;
      v55 = HIBYTE(v52);
    }
    else
    {
      v55 = 0;
      LOBYTE(v54) = 0;
    }
    v54 = (unsigned __int8)v54;
    BYTE1(v54) = v55;
    v106 = v54;
    v56 = *(_QWORD *)(a1 - 64);
    v57 = *(_BYTE *)a1 == 56;
    v125 = 257;
    if ( v57 )
      v58 = sub_B504D0(27, v56, a2, (__int64)&v123, 0, 0);
    else
      v58 = sub_B504D0(26, v56, a2, (__int64)&v123, 0, 0);
    *v20 = v58;
    sub_B44150((_QWORD *)*v20, v112, v53, v106);
    v59 = *v20;
    v60 = *(_QWORD *)(a1 + 48);
    v123 = (unsigned __int64 *)v60;
    if ( v60 )
    {
      sub_B96E90((__int64)&v123, v60, 1);
      v61 = v59 + 48;
      if ( (unsigned __int64 **)(v59 + 48) == &v123 )
      {
        if ( v123 )
          sub_B91220((__int64)&v123, (__int64)v123);
        goto LABEL_81;
      }
      v63 = *(_QWORD *)(v59 + 48);
      if ( !v63 )
      {
LABEL_92:
        v64 = (unsigned __int8 *)v123;
        *(_QWORD *)(v59 + 48) = v123;
        if ( v64 )
          sub_B976B0((__int64)&v123, v64, v61);
        goto LABEL_81;
      }
LABEL_91:
      v107 = v61;
      sub_B91220(v61, v63);
      v61 = v107;
      goto LABEL_92;
    }
    v61 = v59 + 48;
    if ( (unsigned __int64 **)(v59 + 48) != &v123 )
    {
      v63 = *(_QWORD *)(v59 + 48);
      if ( v63 )
        goto LABEL_91;
    }
LABEL_81:
    v19 = *v20;
    v105 = 1;
    if ( *(_QWORD *)v11 )
    {
      v21 = *(_QWORD *)(v11 + 8);
      **(_QWORD **)(v11 + 16) = v21;
      if ( v21 )
LABEL_17:
        *(_QWORD *)(v21 + 16) = *(_QWORD *)(v11 + 16);
    }
    *(_QWORD *)v11 = v19;
    if ( !v19 )
      goto LABEL_8;
LABEL_19:
    v22 = *(_QWORD *)(v19 + 16);
    *(_QWORD *)(v11 + 8) = v22;
    if ( v22 )
      *(_QWORD *)(v22 + 16) = v11 + 8;
    *(_QWORD *)(v11 + 16) = v19 + 16;
    *(_QWORD *)(v19 + 16) = v11;
    if ( v8 )
      continue;
    break;
  }
LABEL_22:
  if ( !*(_QWORD *)(a1 + 16) )
    goto LABEL_97;
LABEL_23:
  sub_C7D6A0(v116, 16LL * v118, 8);
  return v105;
}
