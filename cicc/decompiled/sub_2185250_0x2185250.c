// Function: sub_2185250
// Address: 0x2185250
//
__int64 __fastcall sub_2185250(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 result; // rax
  __int64 v11; // rsi
  _QWORD *v12; // r13
  _QWORD *v13; // r12
  int *v14; // r15
  __int64 v15; // rbx
  __int64 *v16; // rcx
  __int64 v17; // rsi
  unsigned int v18; // ecx
  __int64 v19; // rax
  __int64 v20; // rcx
  _QWORD *v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rdi
  unsigned int v24; // edx
  __int64 *v25; // rcx
  __int64 v26; // r9
  char v27; // r8
  __int64 *v28; // rax
  int v29; // edx
  __int64 v30; // rbx
  unsigned int v31; // esi
  __int64 v32; // rdx
  __int64 *v33; // rbx
  __int64 *v34; // r13
  __int64 v35; // r11
  int v36; // r9d
  int v37; // r8d
  unsigned int v38; // edx
  __int64 *v39; // rdi
  __int64 v40; // rcx
  __int64 v41; // rax
  __int64 v42; // rsi
  __int64 v43; // rdx
  int v44; // r9d
  __int64 v45; // rdi
  int v46; // r15d
  __int64 v47; // rax
  unsigned int v48; // r14d
  __int64 *v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r10
  __int64 v52; // rax
  __int64 v53; // rcx
  __int64 v54; // rdi
  unsigned int v55; // r9d
  __int64 *v56; // rsi
  __int64 v57; // r8
  int v58; // esi
  __int64 *v59; // r10
  int v60; // ecx
  __int64 *v61; // r15
  int v62; // r14d
  __int64 *v63; // r10
  int v64; // eax
  int v65; // eax
  __int64 v66; // rbx
  unsigned __int64 v67; // r15
  __int64 v68; // rdx
  unsigned int v69; // esi
  __int64 v70; // rcx
  __int64 v71; // r8
  unsigned int v72; // edx
  unsigned __int64 v73; // rax
  __int64 v74; // rdi
  __int64 *v75; // r12
  __int64 *v76; // rbx
  int v77; // r9d
  __int64 v78; // r8
  int v79; // r11d
  __int64 *v80; // r10
  unsigned int v81; // edx
  __int64 *v82; // rdi
  __int64 v83; // rcx
  __int64 v84; // rax
  unsigned int v85; // esi
  int v86; // ecx
  __int64 v87; // rax
  int v88; // edi
  __int64 v89; // r14
  __int64 v90; // rax
  __int64 *v91; // rbx
  __int64 *v92; // r15
  __int64 v93; // r14
  unsigned int v94; // edx
  __int64 *v95; // rdi
  __int64 v96; // r10
  __int64 *v97; // rsi
  __int64 v98; // rdx
  __int64 v99; // rcx
  int v100; // r8d
  int v101; // r9d
  int v102; // edi
  int v103; // r8d
  int v104; // esi
  int v105; // r14d
  int v106; // r15d
  int v107; // r11d
  unsigned __int64 v108; // r10
  int v109; // edi
  int i; // ecx
  int v111; // r10d
  __int64 v112; // rdi
  __int64 *v113; // rbx
  __int64 v115; // [rsp+10h] [rbp-120h]
  __int64 v117; // [rsp+20h] [rbp-110h]
  int v118; // [rsp+20h] [rbp-110h]
  __int64 *v119; // [rsp+20h] [rbp-110h]
  __int64 v120; // [rsp+20h] [rbp-110h]
  __int64 v121; // [rsp+20h] [rbp-110h]
  int v125; // [rsp+60h] [rbp-D0h]
  char v126; // [rsp+68h] [rbp-C8h]
  __int64 v127; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v128; // [rsp+78h] [rbp-B8h] BYREF
  __int64 v129; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v130; // [rsp+88h] [rbp-A8h]
  __int64 v131; // [rsp+90h] [rbp-A0h]
  __int64 v132; // [rsp+98h] [rbp-98h]
  __int64 v133[6]; // [rsp+A0h] [rbp-90h] BYREF
  __int64 *v134; // [rsp+D0h] [rbp-60h] BYREF
  __int64 v135; // [rsp+D8h] [rbp-58h]
  _BYTE v136[80]; // [rsp+E0h] [rbp-50h] BYREF

  result = *(unsigned int *)(a2 + 8);
  if ( !(_DWORD)result )
    return result;
  do
  {
    v11 = *(_QWORD *)(*(_QWORD *)a2 + 8LL * (unsigned int)result - 8);
    *(_DWORD *)(a2 + 8) = result - 1;
    v127 = v11;
    v12 = *(_QWORD **)sub_1C01EA0(a3, v11);
    v126 = 0;
    v125 = 0;
    v13 = *(_QWORD **)(sub_1C01EA0(a3, v127) + 8);
    if ( *(_QWORD *)a5 != *(_QWORD *)a5 + 4LL * *(unsigned int *)(a5 + 8) )
    {
      v14 = *(int **)a5;
      v15 = *(_QWORD *)a5 + 4LL * *(unsigned int *)(a5 + 8);
      while ( 1 )
      {
        v18 = sub_217F620(a3, *v14);
        v19 = 1LL << v18;
        v20 = 8LL * (v18 >> 6);
        v21 = (_QWORD *)(v20 + *v12);
        if ( (*v21 & v19) != 0 )
        {
          *v21 &= ~v19;
          v16 = (__int64 *)(*v13 + v20);
          v17 = *v16;
          if ( (*v16 & v19) != 0 )
            goto LABEL_8;
LABEL_5:
          if ( ++v14 == (int *)v15 )
            break;
        }
        else
        {
          v16 = (__int64 *)(*v13 + v20);
          v17 = *v16;
          if ( (*v16 & v19) == 0 )
            goto LABEL_5;
LABEL_8:
          ++v14;
          --v125;
          v126 = 1;
          *v16 = v17 & ~v19;
          if ( v14 == (int *)v15 )
            break;
        }
      }
    }
    v22 = *(unsigned int *)(a9 + 24);
    if ( !(_DWORD)v22 )
      goto LABEL_54;
    v23 = *(_QWORD *)(a9 + 8);
    v24 = (v22 - 1) & (((unsigned int)v127 >> 9) ^ ((unsigned int)v127 >> 4));
    v25 = (__int64 *)(v23 + 56LL * v24);
    v26 = *v25;
    if ( v127 != *v25 )
    {
      for ( i = 1; ; i = v111 )
      {
        if ( v26 == -8 )
          goto LABEL_54;
        v111 = i + 1;
        v24 = (v22 - 1) & (i + v24);
        v25 = (__int64 *)(v23 + 56LL * v24);
        v26 = *v25;
        if ( v127 == *v25 )
          break;
      }
    }
    if ( v25 == (__int64 *)(v23 + 56 * v22) )
      goto LABEL_54;
    v27 = sub_217F4B0(a9, &v127, &v134);
    v28 = v134;
    if ( v27 )
      goto LABEL_18;
    ++*(_QWORD *)a9;
    v29 = *(_DWORD *)(a9 + 16) + 1;
    v30 = a9;
    v31 = *(_DWORD *)(a9 + 24);
    if ( 4 * v29 >= 3 * v31 )
    {
      v31 *= 2;
LABEL_127:
      sub_2183150(v30, v31);
      sub_217F4B0(a9, &v127, &v134);
      v28 = v134;
      v29 = *(_DWORD *)(a9 + 16) + 1;
      goto LABEL_15;
    }
    v30 = a9;
    if ( v31 - *(_DWORD *)(a9 + 20) - v29 <= v31 >> 3 )
      goto LABEL_127;
LABEL_15:
    *(_DWORD *)(a9 + 16) = v29;
    if ( *v28 != -8 )
      --*(_DWORD *)(a9 + 20);
    v32 = v127;
    v28[2] = 0x400000000LL;
    *v28 = v32;
    v28[1] = (__int64)(v28 + 3);
LABEL_18:
    v131 = 0;
    v134 = (__int64 *)v136;
    v132 = 0;
    v135 = 0x400000000LL;
    v130 = 0;
    v33 = (__int64 *)v28[1];
    v129 = 0;
    if ( v33 == &v33[*((unsigned int *)v28 + 4)] )
      goto LABEL_48;
    v34 = &v33[*((unsigned int *)v28 + 4)];
    v115 = a3;
    v35 = a6;
    do
    {
      while ( 1 )
      {
        v52 = *v33;
        v53 = *(unsigned int *)(a4 + 24);
        v128 = *v33;
        if ( !(_DWORD)v53 )
          goto LABEL_26;
        v54 = *(_QWORD *)(a4 + 8);
        v55 = (v53 - 1) & (((unsigned int)v52 >> 4) ^ ((unsigned int)v52 >> 9));
        v56 = (__int64 *)(v54 + 8LL * v55);
        v57 = *v56;
        if ( v52 != *v56 )
        {
          v104 = 1;
          while ( v57 != -8 )
          {
            v105 = v104 + 1;
            v55 = (v53 - 1) & (v104 + v55);
            v56 = (__int64 *)(v54 + 8LL * v55);
            v57 = *v56;
            if ( v52 == *v56 )
              goto LABEL_29;
            v104 = v105;
          }
          goto LABEL_26;
        }
LABEL_29:
        if ( v56 == (__int64 *)(v54 + 8 * v53) )
          goto LABEL_26;
        v58 = v132;
        if ( !(_DWORD)v132 )
        {
          ++v129;
          goto LABEL_32;
        }
        v36 = v132 - 1;
        v37 = v130;
        v38 = (v132 - 1) & (((unsigned int)v52 >> 4) ^ ((unsigned int)v52 >> 9));
        v39 = (__int64 *)(v130 + 8LL * v38);
        v40 = *v39;
        if ( v52 != *v39 )
        {
          v106 = 1;
          v59 = 0;
          while ( v40 != -8 )
          {
            if ( v40 != -16 || v59 )
              v39 = v59;
            v38 = v36 & (v106 + v38);
            v40 = *(_QWORD *)(v130 + 8LL * v38);
            if ( v52 == v40 )
              goto LABEL_21;
            ++v106;
            v59 = v39;
            v39 = (__int64 *)(v130 + 8LL * v38);
          }
          if ( !v59 )
            v59 = v39;
          ++v129;
          v60 = v131 + 1;
          if ( 4 * ((int)v131 + 1) >= (unsigned int)(3 * v132) )
          {
LABEL_32:
            v117 = v35;
            v58 = 2 * v132;
          }
          else
          {
            if ( (int)v132 - HIDWORD(v131) - v60 > (unsigned int)v132 >> 3 )
              goto LABEL_105;
            v117 = v35;
          }
          sub_1E22DE0((__int64)&v129, v58);
          sub_1E1F3B0((__int64)&v129, &v128, v133);
          v59 = (__int64 *)v133[0];
          v52 = v128;
          v35 = v117;
          v60 = v131 + 1;
LABEL_105:
          LODWORD(v131) = v60;
          if ( *v59 != -8 )
            --HIDWORD(v131);
          *v59 = v52;
        }
LABEL_21:
        v41 = (unsigned int)v135;
        if ( (unsigned int)v135 >= HIDWORD(v135) )
        {
          v120 = v35;
          sub_16CD150((__int64)&v134, v136, 0, 8, v37, v36);
          v41 = (unsigned int)v135;
          v35 = v120;
        }
        v134[v41] = v128;
        v42 = *(unsigned int *)(v35 + 24);
        LODWORD(v135) = v135 + 1;
        if ( (_DWORD)v42 )
        {
          v43 = v128;
          v44 = v42 - 1;
          v45 = *(_QWORD *)(v35 + 8);
          v46 = 1;
          LODWORD(v47) = (v42 - 1) & (((unsigned int)v128 >> 9) ^ ((unsigned int)v128 >> 4));
          v48 = v47;
          v49 = (__int64 *)(v45 + 56LL * (unsigned int)v47);
          v50 = *v49;
          v51 = *v49;
          if ( v128 == *v49 )
          {
            if ( v49 == (__int64 *)(v45 + 56 * v42) )
              goto LABEL_26;
LABEL_83:
            v89 = v49[1];
            v90 = *((unsigned int *)v49 + 4);
            if ( v89 + 8 * v90 == v89 )
              goto LABEL_26;
            v119 = v33;
            v91 = (__int64 *)(v89 + 8 * v90);
            v92 = (__int64 *)v49[1];
            v93 = v35;
            while ( 1 )
            {
              if ( !(_DWORD)v132 )
                goto LABEL_89;
              v94 = (v132 - 1) & (((unsigned int)*v92 >> 9) ^ ((unsigned int)*v92 >> 4));
              v95 = (__int64 *)(v130 + 8LL * v94);
              v96 = *v95;
              if ( *v92 != *v95 )
              {
                v102 = 1;
                while ( v96 != -8 )
                {
                  v103 = v102 + 1;
                  v94 = (v132 - 1) & (v102 + v94);
                  v95 = (__int64 *)(v130 + 8LL * v94);
                  v96 = *v95;
                  if ( *v92 == *v95 )
                    goto LABEL_86;
                  v102 = v103;
                }
                goto LABEL_89;
              }
LABEL_86:
              if ( (__int64 *)(v130 + 8LL * (unsigned int)v132) == v95 )
              {
LABEL_89:
                sub_2180CE0((__int64)v133, (__int64)&v129, v92);
                v97 = v92++;
                sub_217E700((__int64)&v134, v97, v98, v99, v100, v101);
                if ( v92 == v91 )
                {
LABEL_90:
                  v33 = v119;
                  v35 = v93;
                  goto LABEL_26;
                }
              }
              else if ( ++v92 == v91 )
              {
                goto LABEL_90;
              }
            }
          }
          while ( 1 )
          {
            if ( v51 == -8 )
              goto LABEL_26;
            v48 = v44 & (v46 + v48);
            v118 = v46 + 1;
            v61 = (__int64 *)(v45 + 56LL * v48);
            v51 = *v61;
            if ( v128 == *v61 )
              break;
            v46 = v118;
          }
          if ( v61 != (__int64 *)(v45 + 56LL * (unsigned int)v42) )
            break;
        }
LABEL_26:
        if ( v34 == ++v33 )
          goto LABEL_47;
      }
      v62 = 1;
      v63 = 0;
      while ( v50 != -8 )
      {
        if ( !v63 && v50 == -16 )
          v63 = v49;
        v47 = v44 & (unsigned int)(v47 + v62);
        v49 = (__int64 *)(v45 + 56 * v47);
        v50 = *v49;
        if ( v128 == *v49 )
          goto LABEL_83;
        ++v62;
      }
      v64 = *(_DWORD *)(v35 + 16);
      if ( !v63 )
        v63 = v49;
      ++*(_QWORD *)v35;
      v65 = v64 + 1;
      if ( 4 * v65 >= (unsigned int)(3 * v42) )
      {
        LODWORD(v42) = 2 * v42;
      }
      else if ( (int)v42 - *(_DWORD *)(v35 + 20) - v65 > (unsigned int)v42 >> 3 )
      {
        goto LABEL_44;
      }
      v121 = v35;
      sub_2182600(v35, v42);
      sub_217F3F0(v121, &v128, v133);
      v35 = v121;
      v63 = (__int64 *)v133[0];
      v43 = v128;
      v65 = *(_DWORD *)(v121 + 16) + 1;
LABEL_44:
      *(_DWORD *)(v35 + 16) = v65;
      if ( *v63 != -8 )
        --*(_DWORD *)(v35 + 20);
      ++v33;
      *v63 = v43;
      v63[1] = (__int64)(v63 + 3);
      v63[2] = 0x400000000LL;
    }
    while ( v34 != v33 );
LABEL_47:
    a3 = v115;
LABEL_48:
    sub_217E1F0(a1, (__int64)&v129, (__int64)&v134);
    if ( (_DWORD)v135 )
    {
      v66 = 8LL * (unsigned int)v135;
      v67 = 0;
      do
      {
        v68 = v134[v67 / 8];
        v67 += 8LL;
        sub_21810D0(a1, a3, v68, v127);
      }
      while ( v66 != v67 );
    }
    if ( v134 != (__int64 *)v136 )
      _libc_free((unsigned __int64)v134);
    j___libc_free_0(v130);
LABEL_54:
    v69 = *(_DWORD *)(a8 + 24);
    if ( v69 )
    {
      v70 = v127;
      v71 = *(_QWORD *)(a8 + 8);
      v72 = (v69 - 1) & (((unsigned int)v127 >> 9) ^ ((unsigned int)v127 >> 4));
      v73 = v71 + 16LL * v72;
      v74 = *(_QWORD *)v73;
      if ( v127 == *(_QWORD *)v73 )
      {
        v125 += *(_DWORD *)(v73 + 8);
        goto LABEL_57;
      }
      v107 = 1;
      v108 = 0;
      while ( v74 != -8 )
      {
        if ( v108 || v74 != -16 )
          v73 = v108;
        v72 = (v69 - 1) & (v107 + v72);
        v113 = (__int64 *)(v71 + 16LL * v72);
        v74 = *v113;
        if ( v127 == *v113 )
        {
          v73 = v71 + 16LL * v72;
          v125 += *((_DWORD *)v113 + 2);
          goto LABEL_57;
        }
        ++v107;
        v108 = v73;
        v73 = v71 + 16LL * v72;
      }
      if ( v108 )
        v73 = v108;
      ++*(_QWORD *)a8;
      v109 = *(_DWORD *)(a8 + 16) + 1;
      if ( 4 * v109 < 3 * v69 )
      {
        if ( v69 - *(_DWORD *)(a8 + 20) - v109 > v69 >> 3 )
          goto LABEL_114;
        v112 = a8;
        goto LABEL_123;
      }
    }
    else
    {
      ++*(_QWORD *)a8;
    }
    v112 = a8;
    v69 *= 2;
LABEL_123:
    sub_1DA35E0(v112, v69);
    sub_217F2A0(a8, &v127, &v134);
    v73 = (unsigned __int64)v134;
    v70 = v127;
    v109 = *(_DWORD *)(a8 + 16) + 1;
LABEL_114:
    *(_DWORD *)(a8 + 16) = v109;
    if ( *(_QWORD *)v73 != -8 )
      --*(_DWORD *)(a8 + 20);
    *(_QWORD *)v73 = v70;
    *(_DWORD *)(v73 + 8) = 0;
LABEL_57:
    *(_DWORD *)(v73 + 8) = v125;
    if ( v126 )
    {
      v75 = *(__int64 **)(v127 + 96);
      v76 = *(__int64 **)(v127 + 88);
      if ( v75 != v76 )
      {
        while ( 2 )
        {
          v84 = *v76;
          v85 = *(_DWORD *)(a7 + 24);
          v133[0] = *v76;
          if ( !v85 )
          {
            ++*(_QWORD *)a7;
            goto LABEL_66;
          }
          v77 = v85 - 1;
          v78 = *(_QWORD *)(a7 + 8);
          v79 = 1;
          v80 = 0;
          v81 = (v85 - 1) & (((unsigned int)v84 >> 9) ^ ((unsigned int)v84 >> 4));
          v82 = (__int64 *)(v78 + 8LL * v81);
          v83 = *v82;
          if ( v84 == *v82 )
          {
LABEL_63:
            if ( v75 == ++v76 )
              goto LABEL_58;
            continue;
          }
          break;
        }
        while ( v83 != -8 )
        {
          if ( v83 != -16 || v80 )
            v82 = v80;
          v81 = v77 & (v79 + v81);
          v83 = *(_QWORD *)(v78 + 8LL * v81);
          if ( v84 == v83 )
            goto LABEL_63;
          ++v79;
          v80 = v82;
          v82 = (__int64 *)(v78 + 8LL * v81);
        }
        if ( !v80 )
          v80 = v82;
        v88 = *(_DWORD *)(a7 + 16);
        ++*(_QWORD *)a7;
        v86 = v88 + 1;
        if ( 4 * (v88 + 1) < 3 * v85 )
        {
          if ( v85 - *(_DWORD *)(a7 + 20) - v86 <= v85 >> 3 )
          {
LABEL_67:
            sub_1DF9CE0(a7, v85);
            sub_1DF93E0(a7, v133, &v134);
            v80 = v134;
            v84 = v133[0];
            v86 = *(_DWORD *)(a7 + 16) + 1;
          }
          *(_DWORD *)(a7 + 16) = v86;
          if ( *v80 != -8 )
            --*(_DWORD *)(a7 + 20);
          *v80 = v84;
          v87 = *(unsigned int *)(a2 + 8);
          if ( (unsigned int)v87 >= *(_DWORD *)(a2 + 12) )
          {
            sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, v78, v77);
            v87 = *(unsigned int *)(a2 + 8);
          }
          *(_QWORD *)(*(_QWORD *)a2 + 8 * v87) = v133[0];
          ++*(_DWORD *)(a2 + 8);
          goto LABEL_63;
        }
LABEL_66:
        v85 *= 2;
        goto LABEL_67;
      }
    }
LABEL_58:
    result = *(unsigned int *)(a2 + 8);
  }
  while ( (_DWORD)result );
  return result;
}
