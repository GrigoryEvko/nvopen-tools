// Function: sub_2D7C490
// Address: 0x2d7c490
//
__int64 __fastcall sub_2D7C490(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v4; // rdi
  __int64 v5; // r9
  __int64 v7; // rax
  __int64 v8; // r12
  unsigned int v9; // esi
  __int64 v10; // r8
  __int64 v11; // r13
  __int64 v12; // r14
  int v13; // r11d
  __int64 *v14; // rdi
  unsigned int v15; // ecx
  __int64 *v16; // rax
  _QWORD *v17; // rax
  __int64 v18; // rdx
  int v19; // eax
  unsigned __int64 v20; // r14
  __int64 v21; // r13
  _BYTE *v22; // r12
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  __int64 v25; // rax
  unsigned int v26; // esi
  __int64 v27; // r8
  int v28; // r11d
  __int64 *v29; // rdi
  unsigned int v30; // ecx
  __int64 *v31; // rdx
  __int64 v32; // r10
  unsigned __int64 v33; // r11
  __int64 v34; // rcx
  unsigned int v35; // edx
  __int64 *v36; // rax
  __int64 v37; // r10
  _BYTE ***v38; // r13
  __int64 v39; // rdx
  _BYTE **v40; // r14
  __int64 v41; // rax
  __int64 v42; // r12
  unsigned int v43; // esi
  _BYTE *v44; // rcx
  int v45; // r11d
  _QWORD *v46; // rdi
  unsigned int v47; // edx
  _QWORD *v48; // rax
  _BYTE *v49; // r8
  unsigned int v50; // esi
  _BYTE *v51; // rdx
  _QWORD *v52; // r10
  int v53; // r11d
  unsigned int v54; // eax
  _QWORD *v55; // rcx
  _BYTE *v56; // rdi
  _BYTE *v57; // rsi
  unsigned int v58; // ecx
  _QWORD *v59; // rdx
  _BYTE *v60; // r9
  unsigned int v61; // eax
  int v62; // edx
  __int64 *v63; // rbx
  __int64 *v64; // r12
  _QWORD *v65; // r13
  __int64 v66; // rdi
  __int64 v67; // rsi
  __int64 v68; // rax
  __int64 *v69; // rdx
  __int64 v70; // rcx
  __int64 v71; // r9
  __int64 v72; // r8
  __int64 v73; // r12
  __int64 v74; // r13
  __int64 v75; // rsi
  __int64 *v76; // rax
  __int64 v77; // rdi
  int v78; // edx
  __int64 *v79; // r8
  int v80; // ecx
  int v81; // r10d
  int v82; // edx
  __int64 *v83; // rdi
  int v84; // eax
  _BYTE ***v85; // [rsp+10h] [rbp-180h]
  __int64 v86; // [rsp+18h] [rbp-178h]
  unsigned __int8 v87; // [rsp+2Fh] [rbp-161h]
  _BYTE *v88; // [rsp+38h] [rbp-158h]
  __int64 v89; // [rsp+48h] [rbp-148h]
  __int64 v90; // [rsp+48h] [rbp-148h]
  __int64 *v91; // [rsp+48h] [rbp-148h]
  _QWORD *v92; // [rsp+48h] [rbp-148h]
  int v93; // [rsp+48h] [rbp-148h]
  __int64 v94; // [rsp+50h] [rbp-140h] BYREF
  _BYTE *v95; // [rsp+58h] [rbp-138h] BYREF
  __int64 v96; // [rsp+60h] [rbp-130h] BYREF
  __int64 v97; // [rsp+68h] [rbp-128h]
  __int64 v98; // [rsp+70h] [rbp-120h]
  unsigned int v99; // [rsp+78h] [rbp-118h]
  __int64 v100; // [rsp+80h] [rbp-110h] BYREF
  __int64 v101; // [rsp+88h] [rbp-108h]
  __int64 v102; // [rsp+90h] [rbp-100h]
  unsigned int v103; // [rsp+98h] [rbp-F8h]
  __int64 v104[6]; // [rsp+A0h] [rbp-F0h] BYREF
  _BYTE *v105; // [rsp+D0h] [rbp-C0h]
  __int64 v106; // [rsp+D8h] [rbp-B8h]
  _BYTE v107[32]; // [rsp+E0h] [rbp-B0h] BYREF
  _BYTE *v108; // [rsp+100h] [rbp-90h] BYREF
  __int64 v109; // [rsp+108h] [rbp-88h]
  _BYTE v110[32]; // [rsp+110h] [rbp-80h] BYREF
  __int64 *v111; // [rsp+130h] [rbp-60h] BYREF
  __int64 v112; // [rsp+138h] [rbp-58h]
  __int64 v113; // [rsp+140h] [rbp-50h]
  __int64 v114; // [rsp+148h] [rbp-48h]
  __int64 *v115; // [rsp+150h] [rbp-40h] BYREF
  __int64 v116; // [rsp+158h] [rbp-38h]
  _BYTE v117[48]; // [rsp+160h] [rbp-30h] BYREF

  v4 = *(_QWORD *)(a1 + 32);
  v105 = v107;
  v106 = 0x400000000LL;
  v87 = sub_DFE6A0(v4);
  if ( !v87 )
    goto LABEL_2;
  v7 = *(_QWORD *)(a2 + 40);
  v94 = a2;
  v8 = *(_QWORD *)(v7 + 56);
  v89 = v7;
  v108 = v110;
  v109 = 0x400000000LL;
  v96 = 0;
  v97 = 0;
  v98 = 0;
  v99 = 0;
  if ( v8 != v7 + 48 )
  {
    v9 = 0;
    v10 = 0;
    v11 = 0;
    v12 = v7 + 48;
    while ( 1 )
    {
      v18 = v8 - 24;
      if ( !v8 )
        v18 = 0;
      v104[0] = v18;
      if ( v9 )
      {
        v13 = 1;
        v14 = 0;
        v15 = (v9 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v16 = (__int64 *)(v10 + 16LL * v15);
        v5 = *v16;
        if ( v18 == *v16 )
        {
LABEL_8:
          v17 = v16 + 1;
          goto LABEL_9;
        }
        while ( v5 != -4096 )
        {
          if ( !v14 && v5 == -8192 )
            v14 = v16;
          v15 = (v9 - 1) & (v13 + v15);
          v16 = (__int64 *)(v10 + 16LL * v15);
          v5 = *v16;
          if ( v18 == *v16 )
            goto LABEL_8;
          ++v13;
        }
        if ( !v14 )
          v14 = v16;
        ++v96;
        v19 = v98 + 1;
        v111 = v14;
        if ( 4 * ((int)v98 + 1) < 3 * v9 )
        {
          if ( v9 - (v19 + HIDWORD(v98)) > v9 >> 3 )
            goto LABEL_86;
          goto LABEL_16;
        }
      }
      else
      {
        ++v96;
        v111 = 0;
      }
      v9 *= 2;
LABEL_16:
      sub_2D71D30((__int64)&v96, v9);
      sub_2D67DF0((__int64)&v96, v104, &v111);
      v18 = v104[0];
      v14 = v111;
      v19 = v98 + 1;
LABEL_86:
      LODWORD(v98) = v19;
      if ( *v14 != -4096 )
        --HIDWORD(v98);
      *v14 = v18;
      v17 = v14 + 1;
      v14[1] = 0;
LABEL_9:
      *v17 = v11;
      v8 = *(_QWORD *)(v8 + 8);
      if ( v8 == v12 )
        break;
      v10 = v97;
      v9 = v99;
      ++v11;
    }
  }
  v20 = (unsigned __int64)v105;
  v21 = v89;
  v22 = &v105[8 * (unsigned int)v106];
  if ( v22 != v105 )
  {
    while ( 1 )
    {
      v79 = (__int64 *)*((_QWORD *)v22 - 1);
      v25 = *v79;
      v104[0] = v25;
      if ( *(_BYTE *)v25 == 84 )
        goto LABEL_22;
      if ( v21 != *(_QWORD *)(v25 + 40) )
      {
        v23 = (unsigned int)v109;
        v24 = (unsigned int)v109 + 1LL;
        if ( v24 > HIDWORD(v109) )
        {
          v91 = v79;
          sub_C8D5F0((__int64)&v108, v110, v24, 8u, (__int64)v79, v5);
          v23 = (unsigned int)v109;
          v79 = v91;
        }
        *(_QWORD *)&v108[8 * v23] = v79;
        LODWORD(v109) = v109 + 1;
        goto LABEL_22;
      }
      v26 = v99;
      if ( !v99 )
        break;
      v5 = v99 - 1;
      v27 = v97;
      v28 = 1;
      v29 = 0;
      v30 = v5 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
      v31 = (__int64 *)(v97 + 16LL * v30);
      v32 = *v31;
      if ( v25 != *v31 )
      {
        while ( v32 != -4096 )
        {
          if ( !v29 && v32 == -8192 )
            v29 = v31;
          v30 = v5 & (v28 + v30);
          v31 = (__int64 *)(v97 + 16LL * v30);
          v32 = *v31;
          if ( v25 == *v31 )
            goto LABEL_27;
          ++v28;
        }
        if ( !v29 )
          v29 = v31;
        ++v96;
        v82 = v98 + 1;
        v111 = v29;
        if ( 4 * ((int)v98 + 1) < 3 * v99 )
        {
          if ( v99 - HIDWORD(v98) - v82 > v99 >> 3 )
          {
LABEL_130:
            LODWORD(v98) = v82;
            if ( *v29 != -4096 )
              --HIDWORD(v98);
            *v29 = v25;
            v29[1] = 0;
            v26 = v99;
            if ( !v99 )
            {
              ++v96;
              v111 = 0;
              goto LABEL_134;
            }
            v27 = v97;
            v5 = v99 - 1;
            v33 = 0;
            goto LABEL_28;
          }
LABEL_141:
          sub_2D71D30((__int64)&v96, v26);
          sub_2D67DF0((__int64)&v96, v104, &v111);
          v25 = v104[0];
          v29 = v111;
          v82 = v98 + 1;
          goto LABEL_130;
        }
LABEL_140:
        v26 = 2 * v99;
        goto LABEL_141;
      }
LABEL_27:
      v33 = v31[1];
LABEL_28:
      v34 = v94;
      v35 = v5 & (((unsigned int)v94 >> 9) ^ ((unsigned int)v94 >> 4));
      v36 = (__int64 *)(v27 + 16LL * v35);
      v37 = *v36;
      if ( v94 != *v36 )
      {
        v93 = 1;
        v83 = 0;
        while ( v37 != -4096 )
        {
          if ( !v83 && v37 == -8192 )
            v83 = v36;
          v35 = v5 & (v93 + v35);
          v36 = (__int64 *)(v27 + 16LL * v35);
          v37 = *v36;
          if ( v94 == *v36 )
            goto LABEL_29;
          ++v93;
        }
        if ( !v83 )
          v83 = v36;
        ++v96;
        v84 = v98 + 1;
        v111 = v83;
        if ( 4 * ((int)v98 + 1) < 3 * v26 )
        {
          if ( v26 - (v84 + HIDWORD(v98)) <= v26 >> 3 )
          {
LABEL_135:
            sub_2D71D30((__int64)&v96, v26);
            sub_2D67DF0((__int64)&v96, &v94, &v111);
            v34 = v94;
            v83 = v111;
            v84 = v98 + 1;
          }
          LODWORD(v98) = v84;
          if ( *v83 != -4096 )
            --HIDWORD(v98);
          *v83 = v34;
          v83[1] = 0;
          goto LABEL_22;
        }
LABEL_134:
        v26 *= 2;
        goto LABEL_135;
      }
LABEL_29:
      if ( v36[1] <= v33 )
      {
LABEL_22:
        v22 -= 8;
        if ( (_BYTE *)v20 == v22 )
          goto LABEL_31;
      }
      else
      {
        v22 -= 8;
        v94 = v104[0];
        if ( (_BYTE *)v20 == v22 )
          goto LABEL_31;
      }
    }
    ++v96;
    v111 = 0;
    goto LABEL_140;
  }
LABEL_31:
  v111 = 0;
  v115 = (__int64 *)v117;
  v112 = 0;
  v113 = 0;
  v114 = 0;
  v116 = 0;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v103 = 0;
  v88 = &v108[8 * (unsigned int)v109];
  if ( v88 == v108 )
  {
    v87 = 0;
    v66 = 0;
    v67 = 0;
    goto LABEL_54;
  }
  v38 = (_BYTE ***)v108;
  while ( 2 )
  {
    while ( 2 )
    {
      v40 = *v38;
      v95 = **v38;
      v41 = sub_B47F80(v95);
      v42 = v41;
      if ( !*(_BYTE *)(a1 + 832)
        || (v68 = sub_986520(v41), v72 = v68 + 32LL * (*(_DWORD *)(v42 + 4) & 0x7FFFFFF), v68 == v72) )
      {
        v43 = v103;
        if ( !v103 )
          goto LABEL_68;
        goto LABEL_37;
      }
      v86 = v42;
      v73 = v68;
      v85 = v38;
      v74 = v72;
      do
      {
        while ( 1 )
        {
          while ( **(_BYTE **)v73 <= 0x1Cu )
          {
LABEL_66:
            v73 += 32;
            if ( v74 == v73 )
              goto LABEL_67;
          }
          v75 = *(_QWORD *)(*(_QWORD *)v73 + 40LL);
          if ( *(_BYTE *)(a1 + 868) )
            break;
LABEL_71:
          v73 += 32;
          sub_C8CC70(a1 + 840, v75, (__int64)v69, v70, v72, v71);
          if ( v74 == v73 )
            goto LABEL_67;
        }
        v76 = *(__int64 **)(a1 + 848);
        v77 = *(unsigned int *)(a1 + 860);
        v69 = &v76[v77];
        if ( v76 != v69 )
        {
          while ( v75 != *v76 )
          {
            if ( v69 == ++v76 )
              goto LABEL_73;
          }
          goto LABEL_66;
        }
LABEL_73:
        if ( (unsigned int)v77 >= *(_DWORD *)(a1 + 856) )
          goto LABEL_71;
        v73 += 32;
        *(_DWORD *)(a1 + 860) = v77 + 1;
        *v69 = v75;
        ++*(_QWORD *)(a1 + 840);
      }
      while ( v74 != v73 );
LABEL_67:
      v43 = v103;
      v42 = v86;
      v38 = v85;
      if ( !v103 )
      {
LABEL_68:
        ++v100;
        v104[0] = 0;
        goto LABEL_69;
      }
LABEL_37:
      v44 = v95;
      v45 = 1;
      v46 = 0;
      v47 = (v43 - 1) & (((unsigned int)v95 >> 9) ^ ((unsigned int)v95 >> 4));
      v48 = (_QWORD *)(v101 + 16LL * v47);
      v49 = (_BYTE *)*v48;
      if ( v95 == (_BYTE *)*v48 )
      {
LABEL_38:
        v48[1] = v42;
        v50 = v114;
        if ( (_DWORD)v114 )
          goto LABEL_39;
LABEL_114:
        v111 = (__int64 *)((char *)v111 + 1);
        v104[0] = 0;
LABEL_115:
        v50 *= 2;
        goto LABEL_116;
      }
      while ( v49 != (_BYTE *)-4096LL )
      {
        if ( v49 == (_BYTE *)-8192LL && !v46 )
          v46 = v48;
        v47 = (v43 - 1) & (v45 + v47);
        v48 = (_QWORD *)(v101 + 16LL * v47);
        v49 = (_BYTE *)*v48;
        if ( v95 == (_BYTE *)*v48 )
          goto LABEL_38;
        ++v45;
      }
      if ( !v46 )
        v46 = v48;
      ++v100;
      v78 = v102 + 1;
      v104[0] = (__int64)v46;
      if ( 4 * ((int)v102 + 1) >= 3 * v43 )
      {
LABEL_69:
        v43 *= 2;
LABEL_70:
        sub_2978470((__int64)&v100, v43);
        sub_2AC1910((__int64)&v100, (__int64 *)&v95, v104);
        v44 = v95;
        v46 = (_QWORD *)v104[0];
        v78 = v102 + 1;
        goto LABEL_111;
      }
      if ( v43 - HIDWORD(v102) - v78 <= v43 >> 3 )
        goto LABEL_70;
LABEL_111:
      LODWORD(v102) = v78;
      if ( *v46 != -4096 )
        --HIDWORD(v102);
      *v46 = v44;
      v46[1] = 0;
      v46[1] = v42;
      v50 = v114;
      if ( !(_DWORD)v114 )
        goto LABEL_114;
LABEL_39:
      v51 = v95;
      v52 = 0;
      v53 = 1;
      v54 = (v50 - 1) & (((unsigned int)v95 >> 9) ^ ((unsigned int)v95 >> 4));
      v55 = (_QWORD *)(v112 + 8LL * v54);
      v56 = (_BYTE *)*v55;
      if ( (_BYTE *)*v55 != v95 )
      {
        while ( v56 != (_BYTE *)-4096LL )
        {
          if ( v56 != (_BYTE *)-8192LL || v52 )
            v55 = v52;
          v54 = (v50 - 1) & (v53 + v54);
          v92 = (_QWORD *)(v112 + 8LL * v54);
          v56 = (_BYTE *)*v92;
          if ( v95 == (_BYTE *)*v92 )
            goto LABEL_40;
          ++v53;
          v52 = v55;
          v55 = (_QWORD *)(v112 + 8LL * v54);
        }
        if ( !v52 )
          v52 = v55;
        v111 = (__int64 *)((char *)v111 + 1);
        v80 = v113 + 1;
        v104[0] = (__int64)v52;
        if ( 4 * ((int)v113 + 1) >= 3 * v50 )
          goto LABEL_115;
        if ( v50 - HIDWORD(v113) - v80 <= v50 >> 3 )
        {
LABEL_116:
          sub_CF4090((__int64)&v111, v50);
          sub_23FDF60((__int64)&v111, (__int64 *)&v95, v104);
          v51 = v95;
          v52 = (_QWORD *)v104[0];
          v80 = v113 + 1;
        }
        LODWORD(v113) = v80;
        if ( *v52 != -4096 )
          --HIDWORD(v113);
        *v52 = v51;
        sub_9C95B0((__int64)&v115, (__int64)v95);
      }
LABEL_40:
      LOWORD(v2) = 0;
      sub_B44220((_QWORD *)v42, v94 + 24, v2);
      v94 = v42;
      sub_BED950((__int64)v104, a1 + 184, v42);
      v57 = v40[3];
      if ( v103 )
      {
        v58 = (v103 - 1) & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
        v59 = (_QWORD *)(v101 + 16LL * v58);
        v60 = (_BYTE *)*v59;
        if ( v57 == (_BYTE *)*v59 )
        {
LABEL_42:
          if ( v59 == (_QWORD *)(v101 + 16LL * v103) )
            break;
          v90 = v59[1];
          v61 = sub_BD2910((__int64)v40);
          if ( (*(_BYTE *)(v90 + 7) & 0x40) != 0 )
            v39 = *(_QWORD *)(v90 - 8);
          else
            v39 = v90 - 32LL * (*(_DWORD *)(v90 + 4) & 0x7FFFFFF);
          ++v38;
          sub_AC2B30(v39 + 32LL * v61, v42);
          if ( v88 == (_BYTE *)v38 )
            goto LABEL_48;
          continue;
        }
        v62 = 1;
        while ( v60 != (_BYTE *)-4096LL )
        {
          v81 = v62 + 1;
          v58 = (v103 - 1) & (v62 + v58);
          v59 = (_QWORD *)(v101 + 16LL * v58);
          v60 = (_BYTE *)*v59;
          if ( v57 == (_BYTE *)*v59 )
            goto LABEL_42;
          v62 = v81;
        }
      }
      break;
    }
    ++v38;
    sub_AC2B30((__int64)v40, v42);
    if ( v88 != (_BYTE *)v38 )
      continue;
    break;
  }
LABEL_48:
  v63 = v115;
  v64 = &v115[(unsigned int)v116];
  if ( v64 != v115 )
  {
    do
    {
      while ( 1 )
      {
        v65 = (_QWORD *)*v63;
        if ( !(unsigned __int8)sub_BD3660(*v63, 1) )
          break;
        if ( v64 == ++v63 )
          goto LABEL_53;
      }
      ++v63;
      sub_B43D60(v65);
    }
    while ( v64 != v63 );
  }
LABEL_53:
  v66 = v101;
  v67 = 16LL * v103;
LABEL_54:
  sub_C7D6A0(v66, v67, 8);
  if ( v115 != (__int64 *)v117 )
    _libc_free((unsigned __int64)v115);
  sub_C7D6A0(v112, 8LL * (unsigned int)v114, 8);
  sub_C7D6A0(v97, 16LL * v99, 8);
  if ( v108 != v110 )
    _libc_free((unsigned __int64)v108);
LABEL_2:
  if ( v105 != v107 )
    _libc_free((unsigned __int64)v105);
  return v87;
}
