// Function: sub_10860F0
// Address: 0x10860f0
//
__int64 __fastcall sub_10860F0(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE *v5; // r13
  unsigned __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rcx
  _QWORD *v9; // r12
  _BYTE *v10; // rdi
  __int64 *v11; // r14
  __int64 v12; // r12
  __int64 v13; // rax
  unsigned int v14; // esi
  __int64 v15; // r13
  __int64 v16; // r14
  __int64 v17; // rdi
  __int64 v18; // r8
  __int64 v19; // r11
  _QWORD *v20; // rax
  __int64 v21; // rcx
  __int64 *v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rax
  unsigned __int64 v25; // rax
  __int64 v26; // rdx
  int v27; // edx
  char v28; // cl
  unsigned __int64 v29; // rax
  int *v30; // rax
  unsigned int v31; // esi
  int v32; // r14d
  __int64 v33; // r9
  __int64 *v34; // rdx
  unsigned int v35; // edi
  __int64 *v36; // rax
  __int64 v37; // rcx
  __int64 result; // rax
  __int64 v39; // r9
  __int64 v40; // rax
  __int64 v41; // rax
  int v42; // r15d
  __int64 v43; // r13
  __int64 *v44; // rdi
  int v45; // eax
  __int64 v46; // rdx
  int v47; // r14d
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // rdx
  __int64 v52; // rax
  _QWORD *v53; // rdi
  __int64 v54; // rdi
  _QWORD *v55; // rdi
  int v56; // eax
  int v57; // ecx
  __int64 v58; // rsi
  __int64 v59; // rax
  __int64 v60; // rdx
  bool v61; // cf
  unsigned __int64 v62; // rax
  __int64 v63; // rdx
  _QWORD *v64; // r13
  __int64 *v65; // r12
  __int64 *v66; // rbx
  __int64 v67; // rcx
  _QWORD *v68; // r14
  _QWORD *v69; // rdi
  __int64 v70; // rdi
  _QWORD *v71; // rdi
  __int64 *v72; // rdi
  _QWORD *v73; // rdx
  int v74; // eax
  int v75; // eax
  int v76; // eax
  int v77; // esi
  __int64 v78; // rdi
  unsigned int v79; // eax
  __int64 v80; // r9
  int v81; // r11d
  __int64 *v82; // r10
  int v83; // esi
  int v84; // esi
  __int64 v85; // r9
  unsigned int v86; // ecx
  __int64 v87; // rdi
  int v88; // r11d
  _QWORD *v89; // r10
  int v90; // ecx
  int v91; // ecx
  _QWORD *v92; // r9
  __int64 v93; // rdi
  int v94; // r10d
  __int64 v95; // rsi
  int v96; // eax
  int v97; // eax
  __int64 v98; // rdi
  __int64 *v99; // r9
  unsigned int v100; // r13d
  int v101; // r10d
  __int64 v102; // rsi
  __int64 v103; // r13
  __int64 v104; // rax
  __int64 v105; // [rsp+8h] [rbp-128h]
  __int64 v106; // [rsp+20h] [rbp-110h]
  __int64 v107; // [rsp+20h] [rbp-110h]
  _QWORD *v108; // [rsp+28h] [rbp-108h]
  __int64 v109; // [rsp+30h] [rbp-100h]
  __int64 *v110; // [rsp+38h] [rbp-F8h]
  __int64 v111; // [rsp+38h] [rbp-F8h]
  __int64 v112; // [rsp+40h] [rbp-F0h]
  __int64 v113; // [rsp+40h] [rbp-F0h]
  _QWORD *v114; // [rsp+40h] [rbp-F0h]
  int v115; // [rsp+40h] [rbp-F0h]
  __int64 *v116; // [rsp+48h] [rbp-E8h]
  unsigned int v117; // [rsp+48h] [rbp-E8h]
  __int64 v118[2]; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v119; // [rsp+60h] [rbp-D0h] BYREF
  _QWORD v120[4]; // [rsp+70h] [rbp-C0h] BYREF
  __int16 v121; // [rsp+90h] [rbp-A0h]
  _QWORD v122[4]; // [rsp+A0h] [rbp-90h] BYREF
  __int16 v123; // [rsp+C0h] [rbp-70h]
  void *v124[2]; // [rsp+D0h] [rbp-60h] BYREF
  int v125; // [rsp+E0h] [rbp-50h]
  __int16 v126; // [rsp+F0h] [rbp-40h]

  v116 = (__int64 *)a2;
  v5 = *(_BYTE **)(a3 + 128);
  v6 = *(_QWORD *)(a3 + 136);
  v7 = sub_22077B0(144);
  v8 = v7;
  v9 = (_QWORD *)v7;
  if ( v7 )
  {
    v10 = (_BYTE *)(v7 + 56);
    *(_QWORD *)(v7 + 32) = 0;
    *(_QWORD *)(v7 + 40) = v7 + 56;
    *(_OWORD *)v7 = 0;
    *(_OWORD *)(v7 + 16) = 0;
    if ( &v5[v6] && !v5 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v124[0] = (void *)v6;
    if ( v6 > 0xF )
    {
      v112 = v7;
      v52 = sub_22409D0(v7 + 40, v124, 0);
      v8 = v112;
      v10 = (_BYTE *)v52;
      *(_QWORD *)(v112 + 40) = v52;
      *(void **)(v112 + 56) = v124[0];
    }
    else
    {
      if ( v6 == 1 )
      {
        *(_BYTE *)(v7 + 56) = *v5;
LABEL_7:
        *(_QWORD *)(v8 + 48) = v6;
        v10[v6] = 0;
        *(_QWORD *)(v8 + 120) = v8 + 136;
        *(_DWORD *)(v8 + 72) = 0;
        *(_QWORD *)(v8 + 80) = 0;
        *(_QWORD *)(v8 + 88) = 0;
        *(_QWORD *)(v8 + 96) = 0;
        *(_QWORD *)(v8 + 104) = 0;
        *(_QWORD *)(v8 + 112) = 0;
        *(_QWORD *)(v8 + 128) = 0x100000000LL;
        goto LABEL_8;
      }
      if ( !v6 )
        goto LABEL_7;
    }
    a2 = (__int64)v5;
    v113 = v8;
    memcpy(v10, v5, v6);
    v8 = v113;
    v6 = (unsigned __int64)v124[0];
    v10 = *(_BYTE **)(v113 + 40);
    goto LABEL_7;
  }
LABEL_8:
  v11 = *(__int64 **)(a1 + 56);
  if ( v11 == *(__int64 **)(a1 + 64) )
  {
    v58 = (__int64)v11 - *(_QWORD *)(a1 + 48);
    v110 = *(__int64 **)(a1 + 48);
    v59 = v58 >> 3;
    if ( v58 >> 3 == 0xFFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"vector::_M_realloc_insert");
    v60 = 1;
    if ( v59 )
      v60 = v58 >> 3;
    v61 = __CFADD__(v60, v59);
    v62 = v60 + v59;
    if ( v61 )
    {
      v103 = 0x7FFFFFFFFFFFFFF8LL;
    }
    else
    {
      if ( !v62 )
      {
        v109 = 0;
        v63 = 8;
        v114 = 0;
LABEL_97:
        a2 = (__int64)v114 + v58;
        if ( a2 )
        {
          *(_QWORD *)a2 = v8;
          v9 = 0;
        }
        if ( v11 == v110 )
        {
LABEL_114:
          v72 = v110;
          if ( v110 )
          {
            v111 = v63;
            a2 = *(_QWORD *)(a1 + 64) - (_QWORD)v72;
            j_j___libc_free_0(v72, a2);
            v63 = v111;
          }
          *(_QWORD *)(a1 + 56) = v63;
          *(_QWORD *)(a1 + 48) = v114;
          *(_QWORD *)(a1 + 64) = v109;
          goto LABEL_65;
        }
        v108 = v9;
        v64 = v114;
        v65 = v110;
        v106 = a1;
        v66 = v11;
        while ( 1 )
        {
          v68 = (_QWORD *)*v65;
          if ( v64 )
            break;
          if ( !v68 )
            goto LABEL_102;
          v69 = (_QWORD *)v68[15];
          if ( v69 != v68 + 17 )
            _libc_free(v69, a2);
          v70 = v68[12];
          if ( v70 )
            j_j___libc_free_0(v70, v68[14] - v70);
          v71 = (_QWORD *)v68[5];
          if ( v71 != v68 + 7 )
            j_j___libc_free_0(v71, v68[7] + 1LL);
          a2 = 144;
          ++v65;
          j_j___libc_free_0(v68, 144);
          v67 = 8;
          if ( v66 == v65 )
          {
LABEL_113:
            v9 = v108;
            a1 = v106;
            v63 = (__int64)(v64 + 2);
            goto LABEL_114;
          }
LABEL_103:
          v64 = (_QWORD *)v67;
        }
        *v64 = v68;
        *v65 = 0;
LABEL_102:
        ++v65;
        v67 = (__int64)(v64 + 1);
        if ( v66 == v65 )
          goto LABEL_113;
        goto LABEL_103;
      }
      if ( v62 > 0xFFFFFFFFFFFFFFFLL )
        v62 = 0xFFFFFFFFFFFFFFFLL;
      v103 = 8 * v62;
    }
    v107 = v8;
    v104 = sub_22077B0(v103);
    v8 = v107;
    v114 = (_QWORD *)v104;
    v63 = v104 + 8;
    v109 = v104 + v103;
    goto LABEL_97;
  }
  if ( v11 )
  {
    *v11 = v8;
    *(_QWORD *)(a1 + 56) += 8LL;
    goto LABEL_11;
  }
  *(_QWORD *)(a1 + 56) = 8;
LABEL_65:
  if ( v9 )
  {
    v53 = (_QWORD *)v9[15];
    if ( v53 != v9 + 17 )
      _libc_free(v53, a2);
    v54 = v9[12];
    if ( v54 )
      j_j___libc_free_0(v54, v9[14] - v54);
    v55 = (_QWORD *)v9[5];
    if ( v55 != v9 + 7 )
      j_j___libc_free_0(v55, v9[7] + 1LL);
    j_j___libc_free_0(v9, 144);
  }
LABEL_11:
  v12 = *(_QWORD *)(*(_QWORD *)(a1 + 56) - 8LL);
  v13 = sub_1084C60((_QWORD *)a1, *(_QWORD *)(a3 + 128), *(_QWORD *)(a3 + 136));
  *(_QWORD *)(v12 + 88) = v13;
  v14 = *(_DWORD *)(a1 + 200);
  v15 = v13;
  v16 = *(_QWORD *)(a3 + 16);
  if ( !v14 )
  {
    ++*(_QWORD *)(a1 + 176);
    goto LABEL_138;
  }
  v17 = *(_QWORD *)(a1 + 184);
  v18 = ((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4);
  v19 = (v14 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
  v20 = (_QWORD *)(v17 + 16 * v19);
  v21 = *v20;
  if ( v16 == *v20 )
  {
LABEL_13:
    v22 = v20 + 1;
    goto LABEL_14;
  }
  v115 = 1;
  v73 = 0;
  while ( v21 != -4096 )
  {
    if ( !v73 && v21 == -8192 )
      v73 = v20;
    LODWORD(v19) = (v14 - 1) & (v115 + v19);
    v20 = (_QWORD *)(v17 + 16LL * (unsigned int)v19);
    v21 = *v20;
    if ( v16 == *v20 )
      goto LABEL_13;
    ++v115;
  }
  if ( !v73 )
    v73 = v20;
  v74 = *(_DWORD *)(a1 + 192);
  ++*(_QWORD *)(a1 + 176);
  v75 = v74 + 1;
  if ( 4 * v75 >= 3 * v14 )
  {
LABEL_138:
    sub_1085690(a1 + 176, 2 * v14);
    v83 = *(_DWORD *)(a1 + 200);
    if ( v83 )
    {
      v84 = v83 - 1;
      v85 = *(_QWORD *)(a1 + 184);
      v86 = v84 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v75 = *(_DWORD *)(a1 + 192) + 1;
      v73 = (_QWORD *)(v85 + 16LL * v86);
      v87 = *v73;
      if ( v16 != *v73 )
      {
        v88 = 1;
        v89 = 0;
        while ( v87 != -4096 )
        {
          if ( !v89 && v87 == -8192 )
            v89 = v73;
          v18 = (unsigned int)(v88 + 1);
          v86 = v84 & (v88 + v86);
          v73 = (_QWORD *)(v85 + 16LL * v86);
          v87 = *v73;
          if ( v16 == *v73 )
            goto LABEL_125;
          ++v88;
        }
        if ( v89 )
          v73 = v89;
      }
      goto LABEL_125;
    }
    goto LABEL_192;
  }
  if ( v14 - *(_DWORD *)(a1 + 196) - v75 <= v14 >> 3 )
  {
    sub_1085690(a1 + 176, v14);
    v90 = *(_DWORD *)(a1 + 200);
    if ( v90 )
    {
      v91 = v90 - 1;
      v92 = 0;
      v93 = *(_QWORD *)(a1 + 184);
      v94 = 1;
      v18 = v91 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v75 = *(_DWORD *)(a1 + 192) + 1;
      v73 = (_QWORD *)(v93 + 16 * v18);
      v95 = *v73;
      if ( v16 != *v73 )
      {
        while ( v95 != -4096 )
        {
          if ( !v92 && v95 == -8192 )
            v92 = v73;
          v18 = v91 & (unsigned int)(v94 + v18);
          v73 = (_QWORD *)(v93 + 16LL * (unsigned int)v18);
          v95 = *v73;
          if ( v16 == *v73 )
            goto LABEL_125;
          ++v94;
        }
        if ( v92 )
          v73 = v92;
      }
      goto LABEL_125;
    }
LABEL_192:
    ++*(_DWORD *)(a1 + 192);
    goto LABEL_193;
  }
LABEL_125:
  *(_DWORD *)(a1 + 192) = v75;
  if ( *v73 != -4096 )
    --*(_DWORD *)(a1 + 196);
  *v73 = v16;
  v22 = v73 + 1;
  v73[1] = 0;
LABEL_14:
  *v22 = v15;
  *(_QWORD *)(v15 + 112) = v12;
  *(_BYTE *)(v15 + 18) = 3;
  if ( *(_DWORD *)(a3 + 168) != 5 )
  {
    v23 = *(_QWORD *)(a3 + 160);
    if ( v23 )
    {
      v24 = sub_1085B40(a1, v23);
      if ( *(_QWORD *)(v24 + 112) )
        sub_C64ED0("two sections have the same comdat", 1u);
      *(_QWORD *)(v24 + 112) = v12;
    }
  }
  v25 = *(unsigned int *)(v15 + 72);
  if ( v25 == 1 )
    goto LABEL_21;
  if ( v25 > 1 )
  {
    *(_DWORD *)(v15 + 72) = 1;
LABEL_21:
    v26 = *(_QWORD *)(v15 + 64);
    goto LABEL_22;
  }
  v39 = *(unsigned int *)(v15 + 76);
  if ( !(_DWORD)v39 )
  {
    sub_C8D5F0(v15 + 64, (const void *)(v15 + 80), 1u, 0x18u, v18, v39);
    v25 = *(unsigned int *)(v15 + 72);
  }
  v26 = *(_QWORD *)(v15 + 64);
  v40 = v26 + 24 * v25;
  if ( v40 != v26 + 24 )
  {
    do
    {
      if ( v40 )
      {
        *(_QWORD *)(v40 + 16) = 0;
        *(_OWORD *)v40 = 0;
      }
      v40 += 24;
    }
    while ( v26 + 24 != v40 );
    v26 = *(_QWORD *)(v15 + 64);
  }
  *(_DWORD *)(v15 + 72) = 1;
LABEL_22:
  *(_QWORD *)(v26 + 16) = 0;
  *(_OWORD *)v26 = 0;
  **(_DWORD **)(v15 + 64) = 2;
  *(_BYTE *)(*(_QWORD *)(v15 + 64) + 20LL) = *(_DWORD *)(a3 + 168);
  v27 = *(_DWORD *)(a3 + 148);
  *(_DWORD *)(v12 + 36) = v27;
  v28 = *(_BYTE *)(a3 + 32);
  v29 = 1LL << v28;
  if ( v28 == 6 )
  {
    LODWORD(v30) = 7340032;
  }
  else
  {
    if ( v29 <= 0x40 )
    {
      if ( v29 - 1 <= 0x1F )
      {
        switch ( v29 )
        {
          case 1uLL:
            LODWORD(v30) = 0x100000;
            goto LABEL_29;
          case 2uLL:
            LODWORD(v30) = 0x200000;
            goto LABEL_29;
          case 4uLL:
            LODWORD(v30) = 3145728;
            goto LABEL_29;
          case 8uLL:
            v30 = &dword_400000;
            goto LABEL_29;
          case 0x10uLL:
            LODWORD(v30) = 5242880;
            goto LABEL_29;
          case 0x20uLL:
            LODWORD(v30) = 6291456;
            goto LABEL_29;
          default:
            goto LABEL_193;
        }
      }
      goto LABEL_193;
    }
    if ( v28 == 10 )
    {
      LODWORD(v30) = 11534336;
    }
    else if ( v29 <= 0x400 )
    {
      switch ( v28 )
      {
        case 8:
          LODWORD(v30) = 9437184;
          break;
        case 9:
          v30 = (int *)&loc_A00000;
          break;
        case 7:
          LODWORD(v30) = 0x800000;
          break;
        default:
          goto LABEL_193;
      }
    }
    else
    {
      if ( v28 != 12 )
      {
        if ( v28 == 13 )
        {
          v30 = (int *)&loc_E00000;
          goto LABEL_29;
        }
        if ( v28 == 11 )
        {
          LODWORD(v30) = 12582912;
          goto LABEL_29;
        }
LABEL_193:
        BUG();
      }
      LODWORD(v30) = 13631488;
    }
  }
LABEL_29:
  *(_QWORD *)(v12 + 80) = a3;
  *(_DWORD *)(v12 + 36) = v27 | (unsigned int)v30;
  v31 = *(_DWORD *)(a1 + 168);
  if ( !v31 )
  {
    ++*(_QWORD *)(a1 + 144);
    goto LABEL_130;
  }
  v32 = 1;
  v33 = *(_QWORD *)(a1 + 152);
  v34 = 0;
  v35 = (v31 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v36 = (__int64 *)(v33 + 16LL * v35);
  v37 = *v36;
  if ( a3 == *v36 )
  {
LABEL_31:
    result = (__int64)(v36 + 1);
    goto LABEL_32;
  }
  while ( v37 != -4096 )
  {
    if ( !v34 && v37 == -8192 )
      v34 = v36;
    v35 = (v31 - 1) & (v32 + v35);
    v36 = (__int64 *)(v33 + 16LL * v35);
    v37 = *v36;
    if ( a3 == *v36 )
      goto LABEL_31;
    ++v32;
  }
  if ( !v34 )
    v34 = v36;
  v56 = *(_DWORD *)(a1 + 160);
  ++*(_QWORD *)(a1 + 144);
  v57 = v56 + 1;
  if ( 4 * (v56 + 1) >= 3 * v31 )
  {
LABEL_130:
    sub_10854B0(a1 + 144, 2 * v31);
    v76 = *(_DWORD *)(a1 + 168);
    if ( v76 )
    {
      v77 = v76 - 1;
      v78 = *(_QWORD *)(a1 + 152);
      v79 = (v76 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v57 = *(_DWORD *)(a1 + 160) + 1;
      v34 = (__int64 *)(v78 + 16LL * v79);
      v80 = *v34;
      if ( a3 != *v34 )
      {
        v81 = 1;
        v82 = 0;
        while ( v80 != -4096 )
        {
          if ( !v82 && v80 == -8192 )
            v82 = v34;
          v79 = v77 & (v81 + v79);
          v34 = (__int64 *)(v78 + 16LL * v79);
          v80 = *v34;
          if ( a3 == *v34 )
            goto LABEL_88;
          ++v81;
        }
        if ( v82 )
          v34 = v82;
      }
      goto LABEL_88;
    }
    goto LABEL_194;
  }
  if ( v31 - *(_DWORD *)(a1 + 164) - v57 <= v31 >> 3 )
  {
    sub_10854B0(a1 + 144, v31);
    v96 = *(_DWORD *)(a1 + 168);
    if ( v96 )
    {
      v97 = v96 - 1;
      v98 = *(_QWORD *)(a1 + 152);
      v99 = 0;
      v100 = v97 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v101 = 1;
      v57 = *(_DWORD *)(a1 + 160) + 1;
      v34 = (__int64 *)(v98 + 16LL * v100);
      v102 = *v34;
      if ( a3 != *v34 )
      {
        while ( v102 != -4096 )
        {
          if ( !v99 && v102 == -8192 )
            v99 = v34;
          v100 = v97 & (v101 + v100);
          v34 = (__int64 *)(v98 + 16LL * v100);
          v102 = *v34;
          if ( a3 == *v34 )
            goto LABEL_88;
          ++v101;
        }
        if ( v99 )
          v34 = v99;
      }
      goto LABEL_88;
    }
LABEL_194:
    ++*(_DWORD *)(a1 + 160);
    BUG();
  }
LABEL_88:
  *(_DWORD *)(a1 + 160) = v57;
  if ( *v34 != -4096 )
    --*(_DWORD *)(a1 + 164);
  *v34 = a3;
  result = (__int64)(v34 + 1);
  v34[1] = 0;
LABEL_32:
  *(_QWORD *)result = v12;
  if ( *(_BYTE *)(a1 + 241) )
  {
    result = sub_E5CAC0(v116, a3);
    v117 = result;
    if ( (unsigned int)result > 0x100000 )
    {
      v41 = a3;
      v42 = 1;
      v43 = v41;
      do
      {
        v45 = v42;
        v121 = 1283;
        v46 = *(_QWORD *)(v43 + 136);
        v120[0] = "$L";
        v47 = v42++ << 20;
        v122[2] = "_";
        v48 = *(_QWORD *)(v43 + 128);
        v120[3] = v46;
        v122[0] = v120;
        v123 = 770;
        v120[2] = v48;
        v124[0] = v122;
        v125 = v45;
        v126 = 2306;
        sub_CA0F50(v118, v124);
        result = sub_1084C60((_QWORD *)a1, v118[0], v118[1]);
        *(_QWORD *)(result + 112) = v12;
        *(_BYTE *)(result + 18) = 6;
        *(_DWORD *)(result + 8) = v47;
        v51 = *(unsigned int *)(v12 + 128);
        if ( v51 + 1 > (unsigned __int64)*(unsigned int *)(v12 + 132) )
        {
          v105 = result;
          sub_C8D5F0(v12 + 120, (const void *)(v12 + 136), v51 + 1, 8u, v49, v50);
          v51 = *(unsigned int *)(v12 + 128);
          result = v105;
        }
        *(_QWORD *)(*(_QWORD *)(v12 + 120) + 8 * v51) = result;
        v44 = (__int64 *)v118[0];
        ++*(_DWORD *)(v12 + 128);
        if ( v44 != &v119 )
          result = j_j___libc_free_0(v44, v119 + 1);
      }
      while ( v117 > v47 + 0x100000 );
    }
  }
  return result;
}
