// Function: sub_1BFA7B0
// Address: 0x1bfa7b0
//
unsigned __int64 __fastcall sub_1BFA7B0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v4; // r10
  __int64 v6; // rsi
  __int64 v8; // r8
  __int64 v9; // rdi
  unsigned int v10; // r11d
  int v11; // ebx
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // r13
  unsigned __int64 v15; // r13
  __int16 v17; // ax
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rax
  unsigned __int64 v21; // rbx
  unsigned int v22; // ecx
  _QWORD *v23; // rax
  __int64 v24; // rdx
  _QWORD *v25; // r10
  _QWORD *v26; // r13
  unsigned __int64 v27; // r15
  __int64 v28; // rax
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // r13
  unsigned __int64 v31; // r13
  unsigned __int64 v32; // rax
  int v33; // eax
  unsigned int v34; // ecx
  _QWORD *v35; // rax
  __int64 v36; // rdx
  __int64 *v37; // r10
  __int64 *v38; // r15
  __int64 *v39; // r13
  __int64 v40; // rsi
  __int64 v41; // rdx
  unsigned int v42; // esi
  unsigned __int64 v43; // rax
  unsigned int v44; // eax
  unsigned __int64 v45; // rax
  int v46; // eax
  __int64 v47; // rax
  int v48; // r10d
  _QWORD *v49; // r9
  int v50; // edi
  int v51; // esi
  __int64 v52; // rax
  __int64 v53; // rdx
  unsigned int v54; // ecx
  __int64 v55; // r13
  unsigned __int8 v56; // al
  bool v57; // al
  unsigned int v58; // eax
  int v59; // eax
  int v60; // ecx
  __int64 v61; // r9
  unsigned int v62; // edx
  __int64 v63; // rdi
  int v64; // r10d
  _QWORD *v65; // r11
  int v66; // eax
  int v67; // edx
  __int64 v68; // rdi
  _QWORD *v69; // r10
  unsigned int v70; // ebx
  int v71; // r9d
  __int64 v72; // rcx
  int v73; // r9d
  __int64 v74; // rdx
  unsigned int v75; // eax
  unsigned __int64 v76; // r9
  __int64 v77; // rdi
  unsigned int v78; // eax
  unsigned int v79; // eax
  int v80; // edi
  _QWORD *v81; // r9
  int v82; // edi
  int v83; // ecx
  int v84; // r10d
  int v85; // r10d
  __int64 v86; // r11
  unsigned int v87; // edx
  __int64 v88; // r9
  int v89; // edi
  _QWORD *v90; // rsi
  int v91; // r9d
  int v92; // r9d
  __int64 v93; // r10
  _QWORD *v94; // rdx
  __int64 v95; // r13
  int v96; // esi
  __int64 v97; // rdi
  const void **v98; // rsi
  __int64 v99; // rax
  unsigned int v100; // ecx
  unsigned __int64 v101; // rax
  unsigned __int64 v102; // rdx
  __int64 v103; // rax
  int v104; // eax
  _BYTE v105[12]; // [rsp+4h] [rbp-7Ch]
  __int64 v106; // [rsp+10h] [rbp-70h]
  _QWORD *v107; // [rsp+18h] [rbp-68h]
  unsigned int v108; // [rsp+20h] [rbp-60h]
  __int64 v109; // [rsp+20h] [rbp-60h]
  __int64 v110; // [rsp+20h] [rbp-60h]
  unsigned int v113; // [rsp+20h] [rbp-60h]
  __int64 v114; // [rsp+20h] [rbp-60h]
  __int64 v115; // [rsp+28h] [rbp-58h]
  __int64 v116; // [rsp+28h] [rbp-58h]
  __int64 v117; // [rsp+28h] [rbp-58h]
  unsigned __int64 v118; // [rsp+28h] [rbp-58h]
  const void **v119; // [rsp+28h] [rbp-58h]
  __int64 v120; // [rsp+28h] [rbp-58h]
  __int64 v121; // [rsp+28h] [rbp-58h]
  __int64 v122; // [rsp+28h] [rbp-58h]
  __int64 v123; // [rsp+28h] [rbp-58h]
  __int64 v124; // [rsp+28h] [rbp-58h]
  __int64 v125; // [rsp+28h] [rbp-58h]
  __int64 v126; // [rsp+28h] [rbp-58h]
  __int64 v127; // [rsp+28h] [rbp-58h]
  __int64 v128; // [rsp+28h] [rbp-58h]
  __int64 v129; // [rsp+28h] [rbp-58h]
  __int64 v130; // [rsp+28h] [rbp-58h]
  __int64 v131; // [rsp+28h] [rbp-58h]
  unsigned __int64 *v132; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v133; // [rsp+38h] [rbp-48h]
  unsigned __int64 v134; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v135; // [rsp+48h] [rbp-38h]

  if ( !a2 )
    return 0;
  v4 = *(_QWORD *)(a3 + 8);
  v6 = *(unsigned int *)(a3 + 24);
  v8 = a3;
  v9 = v4;
  v10 = v6;
  if ( (_DWORD)v6 )
  {
    v11 = v6 - 1;
    v12 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v13 = (__int64 *)(v4 + 16LL * v12);
    v14 = *v13;
    if ( *v13 == a2 )
    {
LABEL_4:
      if ( v13 != (__int64 *)(v4 + 16LL * (unsigned int)v6) )
        return v13[1];
    }
    else
    {
      v33 = 1;
      while ( v14 != -8 )
      {
        v73 = v33 + 1;
        v12 = v11 & (v33 + v12);
        v13 = (__int64 *)(v4 + 16LL * v12);
        v14 = *v13;
        if ( *v13 == a2 )
          goto LABEL_4;
        v33 = v73;
      }
    }
    if ( a4 > 9 )
    {
      v34 = v11 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v35 = (_QWORD *)(v4 + 16LL * v34);
      v36 = *v35;
      if ( *v35 == a2 )
      {
LABEL_37:
        v35[1] = 0;
        return 0;
      }
      v80 = 1;
      v81 = 0;
      while ( v36 != -8 )
      {
        if ( v36 == -16 && !v81 )
          v81 = v35;
        v34 = v11 & (v80 + v34);
        v35 = (_QWORD *)(v4 + 16LL * v34);
        v36 = *v35;
        if ( *v35 == a2 )
          goto LABEL_37;
        ++v80;
      }
      v82 = *(_DWORD *)(v8 + 16);
      if ( v81 )
        v35 = v81;
      ++*(_QWORD *)v8;
      v83 = v82 + 1;
      if ( 4 * (v82 + 1) < (unsigned int)(3 * v6) )
      {
        if ( (int)v6 - *(_DWORD *)(v8 + 20) - v83 > (unsigned int)v6 >> 3 )
        {
LABEL_119:
          *(_DWORD *)(v8 + 16) = v83;
          if ( *v35 != -8 )
            --*(_DWORD *)(v8 + 20);
          *v35 = a2;
          v35[1] = 0;
          goto LABEL_37;
        }
        v128 = v8;
        sub_1BFA5F0(v8, v6);
        v8 = v128;
        v91 = *(_DWORD *)(v128 + 24);
        if ( v91 )
        {
          v92 = v91 - 1;
          v93 = *(_QWORD *)(v128 + 8);
          v94 = 0;
          LODWORD(v95) = v92 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v96 = 1;
          v83 = *(_DWORD *)(v128 + 16) + 1;
          v35 = (_QWORD *)(v93 + 16LL * (unsigned int)v95);
          v97 = *v35;
          if ( *v35 != a2 )
          {
            while ( v97 != -8 )
            {
              if ( v97 == -16 && !v94 )
                v94 = v35;
              v95 = v92 & (unsigned int)(v95 + v96);
              v35 = (_QWORD *)(v93 + 16 * v95);
              v97 = *v35;
              if ( *v35 == a2 )
                goto LABEL_119;
              ++v96;
            }
            if ( v94 )
              v35 = v94;
          }
          goto LABEL_119;
        }
LABEL_197:
        ++*(_DWORD *)(v8 + 16);
        BUG();
      }
LABEL_125:
      v127 = v8;
      sub_1BFA5F0(v8, 2 * v6);
      v8 = v127;
      v84 = *(_DWORD *)(v127 + 24);
      if ( v84 )
      {
        v85 = v84 - 1;
        v86 = *(_QWORD *)(v127 + 8);
        v87 = v85 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v83 = *(_DWORD *)(v127 + 16) + 1;
        v35 = (_QWORD *)(v86 + 16LL * v87);
        v88 = *v35;
        if ( *v35 != a2 )
        {
          v89 = 1;
          v90 = 0;
          while ( v88 != -8 )
          {
            if ( !v90 && v88 == -16 )
              v90 = v35;
            v87 = v85 & (v89 + v87);
            v35 = (_QWORD *)(v86 + 16LL * v87);
            v88 = *v35;
            if ( *v35 == a2 )
              goto LABEL_119;
            ++v89;
          }
          if ( v90 )
            v35 = v90;
        }
        goto LABEL_119;
      }
      goto LABEL_197;
    }
  }
  else if ( a4 > 9 )
  {
    ++*(_QWORD *)a3;
    goto LABEL_125;
  }
  v17 = *(_WORD *)(a2 + 24);
  if ( (unsigned __int16)(v17 - 1) <= 2u )
  {
    v115 = v8;
    v6 = *(_QWORD *)(a2 + 32);
    v18 = a4 + 1;
    v19 = v8;
LABEL_10:
    v20 = sub_1BFA7B0(a1, v6, v19, v18);
    v8 = v115;
    v21 = v20;
    v4 = *(_QWORD *)(v115 + 8);
    LODWORD(v6) = *(_DWORD *)(v115 + 24);
    goto LABEL_11;
  }
  if ( v17 == 4 )
  {
    v25 = *(_QWORD **)(a2 + 32);
    v107 = &v25[*(_QWORD *)(a2 + 40)];
    if ( v107 != v25 )
    {
      v26 = *(_QWORD **)(a2 + 32);
      v27 = 0;
      v108 = a4 + 1;
      while ( 1 )
      {
        v116 = v8;
        v28 = sub_1BFA7B0(a1, *v26, v8, v108);
        v8 = v116;
        v21 = v28;
        if ( !v28 )
          goto LABEL_56;
        while ( 1 )
        {
          v29 = v27;
          v27 = v21;
          if ( !(v29 % v21) )
            break;
          v21 = v29 % v21;
        }
        if ( v107 == ++v26 )
          goto LABEL_26;
      }
    }
LABEL_13:
    v15 = 0;
    goto LABEL_14;
  }
  if ( v17 != 5 )
  {
    if ( v17 == 10 )
    {
      v121 = v8;
      v52 = sub_1649C60(*(_QWORD *)(a2 - 8));
      v8 = v121;
      v54 = a4;
      v55 = v52;
      if ( v52 )
      {
        v56 = *(_BYTE *)(v52 + 16);
        if ( v56 > 0x10u )
          goto LABEL_108;
        v57 = sub_1593BB0(v55, v6, v53, a4);
        v8 = v121;
        v54 = a4;
        if ( !v57 )
        {
          v56 = *(_BYTE *)(v55 + 16);
          if ( v56 <= 3u )
          {
            v58 = sub_15E4C60(v55);
            v8 = v121;
            v21 = v58;
            v4 = *(_QWORD *)(v121 + 8);
            LODWORD(v6) = *(_DWORD *)(v121 + 24);
            goto LABEL_11;
          }
LABEL_108:
          if ( v56 > 0x17u )
          {
            if ( v56 == 53 )
            {
              v4 = *(_QWORD *)(v8 + 8);
              LODWORD(v6) = *(_DWORD *)(v8 + 24);
              v21 = (unsigned int)(1 << *(_WORD *)(v55 + 18)) >> 1;
              goto LABEL_11;
            }
            if ( v56 == 78 )
            {
              v103 = *(_QWORD *)(v55 - 24);
              if ( !*(_BYTE *)(v103 + 16) && (*(_BYTE *)(v103 + 33) & 0x20) != 0 )
              {
                v104 = *(_DWORD *)(v103 + 36);
                if ( v104 == 4046 || v104 == 4242 )
                {
                  v113 = v54;
                  v115 = v8;
                  v6 = sub_146F1B0(a1, *(_QWORD *)(v55 - 24LL * (*(_DWORD *)(v55 + 20) & 0xFFFFFFF)));
                  v18 = v113 + 1;
                  v19 = v115;
                  goto LABEL_10;
                }
              }
            }
          }
          else if ( v56 == 17 && *(_BYTE *)(*(_QWORD *)v55 + 8LL) == 15 )
          {
            v126 = v8;
            v79 = sub_15E0370(v55);
            v8 = v126;
            v21 = v79;
            goto LABEL_112;
          }
        }
      }
LABEL_56:
      v9 = *(_QWORD *)(v8 + 8);
      v10 = *(_DWORD *)(v8 + 24);
      goto LABEL_13;
    }
    if ( v17 )
    {
      if ( v17 != 7 )
        goto LABEL_13;
      if ( *(_QWORD *)(a2 + 40) != 2 )
        goto LABEL_13;
      v99 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL);
      if ( *(_WORD *)(v99 + 24) )
        goto LABEL_13;
      v130 = v8;
      sub_13A3E40((__int64)&v134, *(_QWORD *)(v99 + 32) + 24LL);
      v8 = v130;
      v100 = a4;
      if ( v135 <= 0x40 )
      {
        v21 = v134;
      }
      else
      {
        v21 = *(_QWORD *)v134;
        j_j___libc_free_0_0(v134);
        v100 = a4;
        v8 = v130;
      }
      if ( v21 )
      {
        v131 = v8;
        v101 = sub_1BFA7B0(a1, **(_QWORD **)(a2 + 32), v8, v100 + 1);
        v8 = v131;
        if ( v101 )
        {
          while ( 1 )
          {
            v102 = v101 % v21;
            v101 = v21;
            if ( !v102 )
              break;
            v21 = v102;
          }
LABEL_26:
          v4 = *(_QWORD *)(v8 + 8);
          LODWORD(v6) = *(_DWORD *)(v8 + 24);
          goto LABEL_27;
        }
      }
      goto LABEL_56;
    }
    v74 = *(_QWORD *)(a2 + 32);
    v75 = *(_DWORD *)(v74 + 32);
    v76 = *(_QWORD *)(v74 + 24);
    v77 = 1LL << ((unsigned __int8)v75 - 1);
    if ( v75 > 0x40 )
    {
      v98 = (const void **)(v74 + 24);
      if ( (*(_QWORD *)(v76 + 8LL * ((v75 - 1) >> 6)) & v77) == 0 )
      {
        v129 = v8;
        v133 = *(_DWORD *)(v74 + 32);
        sub_16A4FD0((__int64)&v132, v98);
        v78 = v133;
        v8 = v129;
        goto LABEL_104;
      }
      v114 = v8;
      v135 = *(_DWORD *)(v74 + 32);
      sub_16A4FD0((__int64)&v134, v98);
      LOBYTE(v75) = v135;
      v8 = v114;
      if ( v135 > 0x40 )
      {
        sub_16A8F40((__int64 *)&v134);
        v8 = v114;
        goto LABEL_103;
      }
    }
    else
    {
      if ( (v77 & v76) == 0 )
      {
        v132 = *(unsigned __int64 **)(v74 + 24);
LABEL_157:
        v21 = (unsigned __int64)v132;
        goto LABEL_11;
      }
      v135 = *(_DWORD *)(v74 + 32);
      v134 = v76;
    }
    v134 = ~v134 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v75);
LABEL_103:
    v124 = v8;
    sub_16A7400((__int64)&v134);
    v78 = v135;
    v8 = v124;
    v133 = v135;
    v132 = (unsigned __int64 *)v134;
LABEL_104:
    if ( v78 > 0x40 )
    {
      v125 = v8;
      v21 = *v132;
      j_j___libc_free_0_0(v132);
      v8 = v125;
      v4 = *(_QWORD *)(v125 + 8);
      LODWORD(v6) = *(_DWORD *)(v125 + 24);
      goto LABEL_11;
    }
    v4 = *(_QWORD *)(v8 + 8);
    LODWORD(v6) = *(_DWORD *)(v8 + 24);
    goto LABEL_157;
  }
  v37 = *(__int64 **)(a2 + 32);
  v38 = &v37[*(_QWORD *)(a2 + 40)];
  if ( v37 == v38 )
    goto LABEL_13;
  *(_DWORD *)&v105[8] = 0;
  v39 = *(__int64 **)(a2 + 32);
  v21 = 0;
  *(_QWORD *)v105 = a4 + 1;
  do
  {
    v40 = *v39;
    if ( *(_WORD *)(*v39 + 24) )
    {
      v120 = v8;
      v47 = sub_1BFA7B0(a1, v40, v8, *(unsigned int *)v105);
      v8 = v120;
      if ( v47 && (_DWORD)v47 && ((unsigned int)v47 & ((_DWORD)v47 - 1)) == 0 )
      {
        if ( *(_QWORD *)&v105[4] )
          v47 *= *(_QWORD *)&v105[4];
        *(_QWORD *)&v105[4] = v47;
      }
    }
    else
    {
      v41 = *(_QWORD *)(v40 + 32);
      v42 = *(_DWORD *)(v41 + 32);
      if ( v42 > 0x40 )
      {
        v106 = v8;
        v110 = v41;
        v119 = (const void **)(v41 + 24);
        v46 = sub_16A5940(v41 + 24);
        v8 = v106;
        if ( v46 != 1 )
          goto LABEL_53;
        if ( (*(_QWORD *)(*(_QWORD *)(v110 + 24) + 8LL * ((v42 - 1) >> 6)) & (1LL << ((unsigned __int8)v42 - 1))) != 0 )
        {
          v135 = v42;
          sub_16A4FD0((__int64)&v134, v119);
          LOBYTE(v42) = v135;
          v8 = v106;
          if ( v135 <= 0x40 )
          {
            v43 = v134;
LABEL_47:
            v134 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v42) & ~v43;
          }
          else
          {
            sub_16A8F40((__int64 *)&v134);
            v8 = v106;
          }
          v117 = v8;
          sub_16A7400((__int64)&v134);
          v44 = v135;
          v8 = v117;
          v133 = v135;
          v132 = (unsigned __int64 *)v134;
        }
        else
        {
          v133 = v42;
          sub_16A4FD0((__int64)&v132, v119);
          v44 = v133;
          v8 = v106;
        }
        if ( v44 <= 0x40 )
        {
LABEL_107:
          v45 = (unsigned __int64)v132;
        }
        else
        {
          v109 = v8;
          v118 = *v132;
          j_j___libc_free_0_0(v132);
          v8 = v109;
          v45 = v118;
        }
        if ( v21 )
          v21 *= v45;
        else
          v21 = v45;
        goto LABEL_53;
      }
      v43 = *(_QWORD *)(v41 + 24);
      if ( v43 && (v43 & (v43 - 1)) == 0 )
      {
        if ( _bittest64((const __int64 *)&v43, v42 - 1) )
        {
          v135 = *(_DWORD *)(v41 + 32);
          goto LABEL_47;
        }
        v132 = *(unsigned __int64 **)(v41 + 24);
        goto LABEL_107;
      }
    }
LABEL_53:
    ++v39;
  }
  while ( v38 != v39 );
  if ( *(_QWORD *)&v105[4] )
  {
    v21 *= *(_QWORD *)&v105[4];
    v4 = *(_QWORD *)(v8 + 8);
    LODWORD(v6) = *(_DWORD *)(v8 + 24);
    goto LABEL_11;
  }
LABEL_112:
  v4 = *(_QWORD *)(v8 + 8);
  LODWORD(v6) = *(_DWORD *)(v8 + 24);
LABEL_11:
  if ( !v21 )
  {
    v9 = v4;
    v10 = v6;
    goto LABEL_13;
  }
LABEL_27:
  v9 = v4;
  v10 = v6;
  v15 = v21;
  if ( (v21 & (v21 - 1)) != 0 )
  {
    v30 = (((v21 >> 1) | v21) >> 2) | (v21 >> 1) | v21;
    v31 = (((v30 >> 4) | v30) >> 8) | (v30 >> 4) | v30;
    v15 = (((((v31 >> 16) | v31) >> 32) | (v31 >> 16) | v31) + 1) >> 1;
    if ( v15 )
    {
      while ( 1 )
      {
        v32 = v21;
        v21 = v15;
        if ( !(v32 % v15) )
          break;
        v15 = v32 % v15;
      }
    }
    else
    {
      v15 = v21;
    }
  }
LABEL_14:
  if ( !v10 )
  {
    ++*(_QWORD *)v8;
    goto LABEL_82;
  }
  v22 = (v10 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v23 = (_QWORD *)(v9 + 16LL * v22);
  v24 = *v23;
  if ( *v23 == a2 )
    goto LABEL_16;
  v48 = 1;
  v49 = 0;
  while ( v24 != -8 )
  {
    if ( !v49 && v24 == -16 )
      v49 = v23;
    v22 = (v10 - 1) & (v48 + v22);
    v23 = (_QWORD *)(v9 + 16LL * v22);
    v24 = *v23;
    if ( *v23 == a2 )
      goto LABEL_16;
    ++v48;
  }
  v50 = *(_DWORD *)(v8 + 16);
  if ( v49 )
    v23 = v49;
  ++*(_QWORD *)v8;
  v51 = v50 + 1;
  if ( 4 * (v50 + 1) >= 3 * v10 )
  {
LABEL_82:
    v122 = v8;
    sub_1BFA5F0(v8, 2 * v10);
    v8 = v122;
    v59 = *(_DWORD *)(v122 + 24);
    if ( v59 )
    {
      v60 = v59 - 1;
      v61 = *(_QWORD *)(v122 + 8);
      v62 = (v59 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v51 = *(_DWORD *)(v122 + 16) + 1;
      v23 = (_QWORD *)(v61 + 16LL * v62);
      v63 = *v23;
      if ( *v23 != a2 )
      {
        v64 = 1;
        v65 = 0;
        while ( v63 != -8 )
        {
          if ( v63 == -16 && !v65 )
            v65 = v23;
          v62 = v60 & (v64 + v62);
          v23 = (_QWORD *)(v61 + 16LL * v62);
          v63 = *v23;
          if ( *v23 == a2 )
            goto LABEL_72;
          ++v64;
        }
        if ( v65 )
          v23 = v65;
      }
      goto LABEL_72;
    }
    goto LABEL_198;
  }
  if ( v10 - (v51 + *(_DWORD *)(v8 + 20)) <= v10 >> 3 )
  {
    v123 = v8;
    sub_1BFA5F0(v8, v10);
    v8 = v123;
    v66 = *(_DWORD *)(v123 + 24);
    if ( v66 )
    {
      v67 = v66 - 1;
      v68 = *(_QWORD *)(v123 + 8);
      v69 = 0;
      v70 = (v66 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v71 = 1;
      v51 = *(_DWORD *)(v123 + 16) + 1;
      v23 = (_QWORD *)(v68 + 16LL * v70);
      v72 = *v23;
      if ( *v23 != a2 )
      {
        while ( v72 != -8 )
        {
          if ( v72 == -16 && !v69 )
            v69 = v23;
          v70 = v67 & (v71 + v70);
          v23 = (_QWORD *)(v68 + 16LL * v70);
          v72 = *v23;
          if ( *v23 == a2 )
            goto LABEL_72;
          ++v71;
        }
        if ( v69 )
          v23 = v69;
      }
      goto LABEL_72;
    }
LABEL_198:
    ++*(_DWORD *)(v8 + 16);
    BUG();
  }
LABEL_72:
  *(_DWORD *)(v8 + 16) = v51;
  if ( *v23 != -8 )
    --*(_DWORD *)(v8 + 20);
  *v23 = a2;
  v23[1] = 0;
LABEL_16:
  v23[1] = v15;
  return v15;
}
