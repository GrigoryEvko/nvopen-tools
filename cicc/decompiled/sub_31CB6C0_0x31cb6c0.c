// Function: sub_31CB6C0
// Address: 0x31cb6c0
//
unsigned __int8 *__fastcall sub_31CB6C0(__int64 *a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 v3; // rsi
  __int64 v4; // r9
  unsigned int v5; // r12d
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  _QWORD *v8; // rcx
  __int64 v9; // r15
  __int64 v10; // r14
  char v11; // al
  unsigned __int8 *v12; // r15
  _BYTE *v13; // rax
  _BYTE *v14; // rdx
  _BYTE *v15; // rax
  _QWORD *v17; // rdi
  unsigned int v18; // esi
  _QWORD *v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rsi
  unsigned int v24; // eax
  __int64 v25; // rdi
  __int64 v26; // rsi
  __int64 v27; // r10
  unsigned int v28; // edx
  __int64 *v29; // rax
  __int64 v30; // r13
  int v31; // eax
  __int64 v32; // rdx
  unsigned __int64 v33; // rsi
  unsigned __int64 v34; // rcx
  __int64 v35; // rax
  __int64 v36; // r8
  const __m128i *v37; // rax
  __m128i *v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rsi
  __int64 v42; // rdi
  unsigned int v43; // edx
  unsigned int v44; // eax
  unsigned int v45; // r14d
  _QWORD *v46; // rbx
  __int64 v47; // r13
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // r12
  int v51; // r12d
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rsi
  int v55; // edx
  _BYTE *v56; // r13
  __int64 v57; // r12
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rcx
  unsigned int v62; // eax
  __int64 v63; // r12
  __int64 v64; // rcx
  __int64 v65; // rsi
  unsigned int v66; // edx
  __int64 *v67; // rax
  __int64 v68; // r9
  int v69; // r15d
  __int64 v70; // rax
  __int64 v71; // rdx
  unsigned __int64 v72; // rdi
  _QWORD *v73; // rcx
  unsigned int v74; // esi
  __int64 v75; // rdi
  unsigned int v76; // edx
  __int64 v77; // rax
  unsigned __int8 *v78; // r12
  int v79; // eax
  __int64 v80; // rdx
  unsigned __int64 v81; // rsi
  unsigned __int64 v82; // rcx
  __int64 v83; // r8
  const __m128i *v84; // rax
  __m128i *v85; // rdx
  __int64 v86; // rdi
  const void *v87; // rsi
  char *v88; // r12
  int v89; // eax
  int v90; // r8d
  int v91; // eax
  int v92; // r8d
  __int64 v93; // rdi
  const void *v94; // rsi
  char *v95; // r12
  int v96; // eax
  int v97; // r8d
  __int64 *v98; // [rsp+8h] [rbp-2D8h]
  __int64 v99; // [rsp+10h] [rbp-2D0h]
  __int64 v100; // [rsp+18h] [rbp-2C8h]
  __int64 v102[2]; // [rsp+40h] [rbp-2A0h] BYREF
  __int64 v103; // [rsp+50h] [rbp-290h] BYREF
  __int64 v104; // [rsp+58h] [rbp-288h]
  __int64 v105; // [rsp+60h] [rbp-280h]
  __int64 v106; // [rsp+68h] [rbp-278h]
  int v107; // [rsp+70h] [rbp-270h] BYREF
  __int64 v108; // [rsp+74h] [rbp-26Ch]
  __int64 v109; // [rsp+80h] [rbp-260h]
  _QWORD *v110; // [rsp+88h] [rbp-258h]
  __int64 v111; // [rsp+90h] [rbp-250h]
  _BYTE *v112; // [rsp+A0h] [rbp-240h] BYREF
  __int64 v113; // [rsp+A8h] [rbp-238h]
  _BYTE v114[560]; // [rsp+B0h] [rbp-230h] BYREF

  v2 = a1;
  v3 = *a1;
  v113 = 0x2000000000LL;
  v102[0] = (__int64)&v103;
  v112 = v114;
  v103 = 0;
  v104 = 0;
  v105 = 0;
  v106 = 0;
  v102[1] = (__int64)&v112;
  sub_31CB410(v102, v3, 0);
  v5 = v113;
  if ( !(_DWORD)v113 )
    goto LABEL_9;
  do
  {
    v6 = v5--;
    v7 = (unsigned __int64)&v112[16 * v6 - 16];
    v8 = *(_QWORD **)v7;
    v9 = *(_QWORD *)(v7 + 8);
    LODWORD(v113) = v5;
    v10 = v8[3];
    v11 = *(_BYTE *)v10;
    if ( *(_BYTE *)v10 <= 0x1Cu )
      goto LABEL_15;
    if ( v11 == 62 )
    {
      v13 = *(_BYTE **)(v10 - 32);
      v14 = (_BYTE *)*v2;
      if ( *v13 == 79 )
      {
        if ( v13 != v14 && *((_BYTE **)v13 - 4) != v14 )
          goto LABEL_15;
      }
      else if ( v13 != v14 )
      {
        goto LABEL_15;
      }
      v15 = *(_BYTE **)(v10 - 64);
      if ( *v15 != 61 )
        goto LABEL_15;
      v56 = (_BYTE *)*((_QWORD *)v15 - 4);
      if ( *v56 != 22 || !(unsigned __int8)sub_B2D680(*((_QWORD *)v15 - 4)) || v2[2] )
        goto LABEL_15;
      v57 = *(_QWORD *)(v10 + 40);
      v58 = *(_QWORD *)(sub_B43CB0(v10) + 80);
      if ( v58 )
        v58 -= 24;
      if ( v58 != v57 )
        goto LABEL_15;
      v59 = *(_QWORD *)(v10 + 40);
      v60 = v2[1];
      if ( v59 )
      {
        v61 = (unsigned int)(*(_DWORD *)(v59 + 44) + 1);
        v62 = *(_DWORD *)(v59 + 44) + 1;
      }
      else
      {
        v61 = 0;
        v62 = 0;
      }
      if ( *(_DWORD *)(v60 + 32) <= v62 )
        goto LABEL_15;
      v63 = *(_QWORD *)(*(_QWORD *)(v60 + 24) + 8 * v61);
      if ( !v63 )
        goto LABEL_15;
      v64 = *(unsigned int *)(a2 + 32);
      v65 = *(_QWORD *)(a2 + 16);
      if ( (_DWORD)v64 )
      {
        v66 = (v64 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v67 = (__int64 *)(v65 + 16LL * v66);
        v68 = *v67;
        if ( v10 == *v67 )
        {
LABEL_74:
          v69 = *((_DWORD *)v67 + 2);
          v70 = sub_22077B0(0x20u);
          if ( v70 )
          {
            v71 = *(_QWORD *)(v63 + 72);
            *(_DWORD *)v70 = v69;
            *(_QWORD *)(v70 + 16) = v56;
            *(_QWORD *)(v70 + 4) = v71;
            *(_QWORD *)(v70 + 24) = v10;
          }
          v72 = v2[2];
          v2[2] = v70;
          if ( v72 )
            j_j___libc_free_0(v72);
          goto LABEL_40;
        }
        v89 = 1;
        while ( v68 != -4096 )
        {
          v90 = v89 + 1;
          v66 = (v64 - 1) & (v89 + v66);
          v67 = (__int64 *)(v65 + 16LL * v66);
          v68 = *v67;
          if ( v10 == *v67 )
            goto LABEL_74;
          v89 = v90;
        }
      }
      v67 = (__int64 *)(v65 + 16 * v64);
      goto LABEL_74;
    }
    if ( (unsigned __int8)(v11 - 78) <= 1u )
      goto LABEL_5;
    if ( v11 == 63 )
    {
      if ( *v8 != *(_QWORD *)(v10 - 32LL * (*(_DWORD *)(v10 + 4) & 0x7FFFFFF)) )
        goto LABEL_15;
LABEL_5:
      if ( !v9 )
        v9 = (__int64)v8;
      sub_31CB410(v102, v8[3], v9);
      v5 = v113;
      continue;
    }
    if ( v11 == 61 )
    {
      v21 = *(_QWORD *)(v10 + 40);
      v22 = v2[1];
      if ( v21 )
      {
        v23 = (unsigned int)(*(_DWORD *)(v21 + 44) + 1);
        v24 = *(_DWORD *)(v21 + 44) + 1;
      }
      else
      {
        v23 = 0;
        v24 = 0;
      }
      if ( v24 >= *(_DWORD *)(v22 + 32) )
        goto LABEL_15;
      v25 = *(_QWORD *)(*(_QWORD *)(v22 + 24) + 8 * v23);
      if ( !v25 )
        goto LABEL_15;
      v26 = *(unsigned int *)(a2 + 32);
      v27 = *(_QWORD *)(a2 + 16);
      if ( (_DWORD)v26 )
      {
        v28 = (v26 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v29 = (__int64 *)(v27 + 16LL * v28);
        v30 = *v29;
        if ( v10 == *v29 )
          goto LABEL_38;
        v91 = 1;
        while ( v30 != -4096 )
        {
          v92 = v91 + 1;
          v28 = (v26 - 1) & (v91 + v28);
          v29 = (__int64 *)(v27 + 16LL * v28);
          v30 = *v29;
          if ( v10 == *v29 )
            goto LABEL_38;
          v91 = v92;
        }
      }
      v29 = (__int64 *)(v27 + 16 * v26);
LABEL_38:
      v31 = *((_DWORD *)v29 + 2);
      v32 = *((unsigned int *)v2 + 8);
      v110 = v8;
      v33 = *((unsigned int *)v2 + 9);
      v34 = v2[3];
      v111 = v9;
      v107 = v31;
      v35 = *(_QWORD *)(v25 + 72);
      v36 = v32 + 1;
      v109 = 0;
      v108 = v35;
      v37 = (const __m128i *)&v107;
      if ( v32 + 1 > v33 )
      {
        v86 = (__int64)(v2 + 3);
        v87 = v2 + 5;
        if ( v34 > (unsigned __int64)&v107 || (unsigned __int64)&v107 >= v34 + 40 * v32 )
        {
          sub_C8D5F0(v86, v87, v32 + 1, 0x28u, v36, v4);
          v34 = v2[3];
          v32 = *((unsigned int *)v2 + 8);
          v37 = (const __m128i *)&v107;
        }
        else
        {
          v88 = (char *)&v107 - v34;
          sub_C8D5F0(v86, v87, v32 + 1, 0x28u, v36, v4);
          v34 = v2[3];
          v32 = *((unsigned int *)v2 + 8);
          v37 = (const __m128i *)&v88[v34];
        }
      }
      v38 = (__m128i *)(v34 + 40 * v32);
      *v38 = _mm_loadu_si128(v37);
      v38[1] = _mm_loadu_si128(v37 + 1);
      v38[2].m128i_i64[0] = v37[2].m128i_i64[0];
      ++*((_DWORD *)v2 + 8);
LABEL_40:
      v5 = v113;
      continue;
    }
    if ( v11 != 85 )
    {
      if ( v11 == 84 )
      {
        v17 = *(_QWORD **)(v10 - 8);
        v18 = *(_DWORD *)(v10 + 4) & 0x7FFFFFF;
        if ( v18 > 1 )
        {
          v19 = v17 + 4;
          v20 = (__int64)&v17[4 * v18];
          while ( *v17 == *v19 )
          {
            v19 += 4;
            if ( v19 == (_QWORD *)v20 )
              goto LABEL_5;
          }
          goto LABEL_15;
        }
        goto LABEL_5;
      }
LABEL_15:
      v12 = (unsigned __int8 *)v10;
      goto LABEL_16;
    }
    v39 = *(_QWORD *)(v10 - 32);
    if ( v39 && !*(_BYTE *)v39 && *(_QWORD *)(v39 + 24) == *(_QWORD *)(v10 + 80) && (*(_BYTE *)(v39 + 33) & 0x20) != 0 )
    {
      if ( sub_B46A10(v8[3]) )
        continue;
      goto LABEL_15;
    }
    v40 = *(_QWORD *)(v10 + 40);
    v41 = v2[1];
    if ( v40 )
    {
      v42 = (unsigned int)(*(_DWORD *)(v40 + 44) + 1);
      v43 = *(_DWORD *)(v40 + 44) + 1;
    }
    else
    {
      v42 = 0;
      v43 = 0;
    }
    if ( v43 >= *(_DWORD *)(v41 + 32) )
      goto LABEL_15;
    v100 = *(_QWORD *)(*(_QWORD *)(v41 + 24) + 8 * v42);
    if ( !v100 || *(_BYTE *)v39 == 25 )
      goto LABEL_15;
    v99 = v9;
    v44 = 0;
    v12 = (unsigned __int8 *)v8[3];
    v98 = v2;
    v45 = 0;
    v46 = v8;
    while ( 1 )
    {
      v47 = -32 - 32LL * v44;
      if ( (v12[7] & 0x80u) == 0 )
        goto LABEL_80;
      v48 = sub_BD2BC0((__int64)v12);
      v50 = v48 + v49;
      if ( (v12[7] & 0x80u) == 0 )
      {
        if ( (unsigned int)(v50 >> 4) )
LABEL_120:
          BUG();
LABEL_80:
        v54 = 0;
        goto LABEL_56;
      }
      if ( !(unsigned int)((v50 - sub_BD2BC0((__int64)v12)) >> 4) )
        goto LABEL_80;
      if ( (v12[7] & 0x80u) == 0 )
        goto LABEL_120;
      v51 = *(_DWORD *)(sub_BD2BC0((__int64)v12) + 8);
      if ( (v12[7] & 0x80u) == 0 )
        BUG();
      v52 = sub_BD2BC0((__int64)v12);
      v54 = 32LL * (unsigned int)(*(_DWORD *)(v52 + v53 - 4) - v51);
LABEL_56:
      if ( v45 >= (unsigned int)((32LL * (*((_DWORD *)v12 + 1) & 0x7FFFFFF) + v47 - v54) >> 5) )
        break;
      if ( *v46 == *(_QWORD *)&v12[32 * (v45 - (unsigned __int64)(*((_DWORD *)v12 + 1) & 0x7FFFFFF))]
        && !(unsigned __int8)sub_B49B80((__int64)v12, v45, 81) )
      {
        goto LABEL_16;
      }
      v55 = *v12;
      ++v45;
      switch ( v55 )
      {
        case '(':
          v44 = sub_B491D0((__int64)v12);
          break;
        case 'U':
          v44 = 0;
          break;
        case '"':
          v44 = 2;
          break;
        default:
          BUG();
      }
    }
    v73 = v46;
    v2 = v98;
    v74 = *(_DWORD *)(a2 + 32);
    v75 = *(_QWORD *)(a2 + 16);
    if ( v74 )
    {
      v76 = (v74 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v77 = v75 + 16LL * v76;
      v78 = *(unsigned __int8 **)v77;
      if ( v12 == *(unsigned __int8 **)v77 )
        goto LABEL_94;
      v96 = 1;
      while ( v78 != (unsigned __int8 *)-4096LL )
      {
        v97 = v96 + 1;
        v76 = (v74 - 1) & (v96 + v76);
        v77 = v75 + 16LL * v76;
        v78 = *(unsigned __int8 **)v77;
        if ( v12 == *(unsigned __int8 **)v77 )
          goto LABEL_94;
        v96 = v97;
      }
    }
    v77 = v75 + 16LL * v74;
LABEL_94:
    v79 = *(_DWORD *)(v77 + 8);
    v80 = *((unsigned int *)v98 + 8);
    v110 = v73;
    v81 = *((unsigned int *)v98 + 9);
    v82 = v98[3];
    v111 = v99;
    v107 = v79;
    v83 = v80 + 1;
    v109 = 0;
    v108 = *(_QWORD *)(v100 + 72);
    v84 = (const __m128i *)&v107;
    if ( v80 + 1 > v81 )
    {
      v93 = (__int64)(v98 + 3);
      v94 = v98 + 5;
      if ( v82 > (unsigned __int64)&v107 || (unsigned __int64)&v107 >= v82 + 40 * v80 )
      {
        sub_C8D5F0(v93, v94, v80 + 1, 0x28u, v83, (__int64)&v107);
        v82 = v98[3];
        v80 = *((unsigned int *)v98 + 8);
        v84 = (const __m128i *)&v107;
      }
      else
      {
        v95 = (char *)&v107 - v82;
        sub_C8D5F0(v93, v94, v80 + 1, 0x28u, v83, (__int64)&v107 - v82);
        v82 = v98[3];
        v80 = *((unsigned int *)v98 + 8);
        v84 = (const __m128i *)&v95[v82];
      }
    }
    v85 = (__m128i *)(v82 + 40 * v80);
    *v85 = _mm_loadu_si128(v84);
    v85[1] = _mm_loadu_si128(v84 + 1);
    v85[2].m128i_i64[0] = v84[2].m128i_i64[0];
    v5 = v113;
    ++*((_DWORD *)v98 + 8);
  }
  while ( v5 );
LABEL_9:
  v12 = 0;
  if ( !v2[2] )
    v12 = (unsigned __int8 *)*v2;
LABEL_16:
  sub_C7D6A0(v104, 8LL * (unsigned int)v106, 8);
  if ( v112 != v114 )
    _libc_free((unsigned __int64)v112);
  return v12;
}
