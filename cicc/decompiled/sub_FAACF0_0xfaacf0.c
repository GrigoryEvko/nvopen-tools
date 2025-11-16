// Function: sub_FAACF0
// Address: 0xfaacf0
//
__int64 __fastcall sub_FAACF0(__int64 a1, __int64 a2)
{
  __int64 v2; // r9
  _QWORD *v3; // r13
  int v4; // eax
  __int64 v5; // rbx
  unsigned int v6; // r13d
  __int64 v7; // rdx
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r11
  __int64 v14; // rcx
  __int64 v15; // rdi
  __int64 v16; // rax
  unsigned int v17; // edx
  __int64 v18; // rax
  __int64 *v19; // rax
  __int64 v20; // r8
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  unsigned __int64 v25; // rax
  int v26; // esi
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // r10
  __int64 v32; // rdx
  __int64 v33; // rdi
  __int64 v34; // rax
  __int64 v35; // r9
  __int64 v36; // rcx
  __int64 *v37; // r15
  __int64 v38; // rdi
  __int64 v39; // rdx
  _QWORD *v40; // rax
  unsigned int v41; // edx
  _QWORD *v42; // rbx
  _QWORD *v43; // rdi
  __int64 v45; // r9
  _QWORD *v46; // rdx
  _QWORD *v47; // r9
  __int64 v48; // rdi
  __int64 v49; // rdx
  __int64 v50; // rsi
  __int64 v51; // r10
  __int64 v52; // rdx
  int v53; // r12d
  __int64 *v54; // rax
  int v55; // ecx
  _QWORD *v56; // rdx
  __int64 *v57; // r8
  __int64 *v58; // rax
  __int64 *v59; // rcx
  char v60; // r9
  unsigned __int64 v61; // r10
  __int64 v62; // rdx
  __int64 v63; // rdi
  char *v64; // rax
  __int64 v65; // rbx
  char *v66; // r11
  __int64 v67; // rdi
  char *v68; // rdi
  unsigned int *v69; // rdi
  unsigned int *v70; // r10
  __int64 v71; // r9
  __int64 v72; // r9
  __int64 v73; // rdx
  __int64 v74; // rax
  __int64 v75; // r9
  __int64 v76; // rdx
  __int64 v77; // r8
  int v78; // r11d
  __int64 *v79; // rdi
  int v80; // r11d
  __int64 v81; // r8
  int v82; // edi
  int v83; // [rsp+4h] [rbp-8Ch]
  __int64 v84; // [rsp+8h] [rbp-88h]
  __int64 v85; // [rsp+8h] [rbp-88h]
  __int64 v86; // [rsp+10h] [rbp-80h]
  int v87; // [rsp+18h] [rbp-78h]
  int v88; // [rsp+18h] [rbp-78h]
  unsigned int v89; // [rsp+18h] [rbp-78h]
  unsigned __int8 v90; // [rsp+1Fh] [rbp-71h]
  __int64 v92; // [rsp+28h] [rbp-68h]
  __int64 v93; // [rsp+38h] [rbp-58h] BYREF
  __int64 v94; // [rsp+40h] [rbp-50h] BYREF
  _QWORD *v95; // [rsp+48h] [rbp-48h]
  __int64 v96; // [rsp+50h] [rbp-40h]
  unsigned int v97; // [rsp+58h] [rbp-38h]

  v2 = 0;
  v3 = 0;
  v4 = *(_DWORD *)(a1 + 4);
  v90 = 0;
  v94 = 0;
  v5 = *(_QWORD *)(a1 + 40);
  v95 = 0;
  v96 = 0;
  v97 = 0;
  v86 = ((v4 & 0x7FFFFFFu) >> 1) - 1;
  v92 = 0;
  if ( (v4 & 0x7FFFFFFu) >> 1 == 1 )
    goto LABEL_49;
  v6 = 3;
  do
  {
    v7 = *(_QWORD *)(a1 - 8);
    v8 = *(_QWORD *)(v7 + 32LL * (v6 - 1));
    v9 = 32LL * v6;
    if ( (_DWORD)v92 == -2 )
      v9 = 32;
    v10 = *(_QWORD *)(v7 + v9);
    v11 = sub_AA5930(v10);
    v13 = v12;
    v14 = v11;
LABEL_6:
    if ( v13 != v14 )
    {
      while ( 1 )
      {
        v15 = *(_QWORD *)(v14 - 8);
        v16 = 0x1FFFFFFFE0LL;
        v17 = *(_DWORD *)(v14 + 4) & 0x7FFFFFF;
        if ( !v17 )
          break;
        v18 = 0;
        a2 = v15 + 32LL * *(unsigned int *)(v14 + 72);
        do
        {
          if ( v5 == *(_QWORD *)(a2 + 8 * v18) )
          {
            v16 = 32 * v18;
            goto LABEL_12;
          }
          ++v18;
        }
        while ( v17 != (_DWORD)v18 );
        v19 = (__int64 *)(v15 + 0x1FFFFFFFE0LL);
        v20 = *(_QWORD *)(v15 + 0x1FFFFFFFE0LL);
        if ( v8 == v20 )
          goto LABEL_51;
LABEL_13:
        v21 = *(_QWORD *)(v14 + 32);
        if ( !v21 )
          BUG();
        v14 = 0;
        if ( *(_BYTE *)(v21 - 24) != 84 )
          goto LABEL_6;
        v14 = v21 - 24;
        if ( v13 == v21 - 24 )
          goto LABEL_16;
      }
LABEL_12:
      v19 = (__int64 *)(v15 + v16);
      v20 = *v19;
      if ( v8 != *v19 )
        goto LABEL_13;
LABEL_51:
      a2 = 32LL * *(unsigned int *)(v14 + 72);
      v45 = a2 + 8LL * v17;
      v46 = (_QWORD *)(v15 + a2);
      v47 = (_QWORD *)(v15 + v45);
      if ( v47 != (_QWORD *)(v15 + a2) )
      {
        a2 = 0;
        do
        {
          v48 = v5 == *v46++;
          a2 += v48;
        }
        while ( v46 != v47 );
        if ( a2 == 1 )
        {
          v49 = **(_QWORD **)(a1 - 8);
          if ( v49 )
          {
            if ( v20 )
            {
              v50 = v19[1];
              *(_QWORD *)v19[2] = v50;
              if ( v50 )
                *(_QWORD *)(v50 + 16) = v19[2];
            }
            *v19 = v49;
            a2 = *(_QWORD *)(v49 + 16);
            v19[1] = a2;
            if ( a2 )
              *(_QWORD *)(a2 + 16) = v19 + 1;
            v19[2] = v49 + 16;
            v90 = 1;
            *(_QWORD *)(v49 + 16) = v19;
          }
          else
          {
            if ( v20 )
            {
              a2 = v19[2];
              v52 = v19[1];
              *(_QWORD *)a2 = v52;
              if ( v52 )
              {
                a2 = v19[2];
                *(_QWORD *)(v52 + 16) = a2;
              }
              *v19 = 0;
            }
            v90 = 1;
          }
        }
      }
      goto LABEL_13;
    }
LABEL_16:
    v22 = sub_AA4FF0(v10);
    v23 = v10 + 48;
    v24 = v22;
    if ( v22 )
    {
      v24 = v22 - 24;
      v25 = *(_QWORD *)(v10 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v25 == v23 )
        goto LABEL_39;
LABEL_18:
      if ( !v25 )
        BUG();
      v26 = *(unsigned __int8 *)(v25 - 24);
      v27 = v25 - 24;
      a2 = (unsigned int)(v26 - 30);
      if ( (unsigned int)a2 >= 0xB )
        v27 = 0;
      if ( v27 != v24 )
        goto LABEL_39;
      goto LABEL_22;
    }
    v25 = *(_QWORD *)(v10 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v25 != v23 )
      goto LABEL_18;
LABEL_22:
    if ( !sub_AA54C0(v10) )
      goto LABEL_39;
    v28 = *(_QWORD *)(v10 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v28 == v10 + 48 )
      goto LABEL_165;
    if ( !v28 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v28 - 24) - 30 > 0xA )
LABEL_165:
      BUG();
    if ( *(_BYTE *)(v28 - 24) != 31 )
      goto LABEL_39;
    if ( (*(_DWORD *)(v28 - 20) & 0x7FFFFFF) != 1 )
      goto LABEL_39;
    v29 = sub_AA5930(*(_QWORD *)(v28 - 56));
    v31 = v30;
    v93 = v29;
    v32 = v29;
    if ( v29 == v31 )
      goto LABEL_39;
    while ( 1 )
    {
      v33 = *(_QWORD *)(v32 - 8);
      if ( (*(_DWORD *)(v32 + 4) & 0x7FFFFFF) == 0 )
      {
LABEL_64:
        v35 = 0xFFFFFFFFLL;
        if ( v8 == *(_QWORD *)(v33 + 0x1FFFFFFFE0LL) )
          break;
        goto LABEL_65;
      }
      v34 = 0;
      a2 = v33 + 32LL * *(unsigned int *)(v32 + 72);
      while ( 1 )
      {
        v35 = (unsigned int)v34;
        if ( v10 == *(_QWORD *)(a2 + 8 * v34) )
          break;
        if ( (*(_DWORD *)(v32 + 4) & 0x7FFFFFF) == (_DWORD)++v34 )
          goto LABEL_64;
      }
      if ( v8 == *(_QWORD *)(v33 + 32 * v34) )
        break;
LABEL_65:
      sub_F8F2F0((__int64)&v93);
      v32 = v93;
      if ( v51 == v93 )
        goto LABEL_39;
    }
    a2 = v97;
    if ( !v97 )
    {
      ++v94;
LABEL_135:
      v88 = v35;
      v84 = v32;
      sub_FAA9D0((__int64)&v94, 2 * v97);
      if ( v97 )
      {
        v32 = v84;
        LODWORD(v35) = v88;
        v55 = v96 + 1;
        a2 = (v97 - 1) & (((unsigned int)v84 >> 9) ^ ((unsigned int)v84 >> 4));
        v54 = &v95[5 * a2];
        v77 = *v54;
        if ( v84 == *v54 )
          goto LABEL_78;
        v78 = 1;
        v79 = 0;
        while ( v77 != -4096 )
        {
          if ( !v79 && v77 == -8192 )
            v79 = v54;
          a2 = (v97 - 1) & ((_DWORD)a2 + v78);
          v54 = &v95[5 * a2];
          v77 = *v54;
          if ( v84 == *v54 )
            goto LABEL_78;
          ++v78;
        }
LABEL_147:
        if ( v79 )
          v54 = v79;
        goto LABEL_78;
      }
LABEL_164:
      LODWORD(v96) = v96 + 1;
      BUG();
    }
    v36 = (v97 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
    v37 = &v95[5 * v36];
    v38 = *v37;
    if ( v32 == *v37 )
    {
LABEL_36:
      v39 = *((unsigned int *)v37 + 4);
      v40 = v37 + 1;
      if ( v39 + 1 > (unsigned __int64)*((unsigned int *)v37 + 5) )
      {
        a2 = (__int64)(v37 + 3);
        v87 = v35;
        sub_C8D5F0((__int64)(v37 + 1), v37 + 3, v39 + 1, 4u, v39 + 1, v35);
        v39 = *((unsigned int *)v37 + 4);
        LODWORD(v35) = v87;
        v40 = v37 + 1;
      }
      goto LABEL_38;
    }
    v53 = 1;
    v54 = 0;
    while ( v38 != -4096 )
    {
      if ( v38 == -8192 && !v54 )
        v54 = v37;
      v82 = v53++;
      LODWORD(v36) = (v97 - 1) & (v82 + v36);
      v37 = &v95[5 * (unsigned int)v36];
      v38 = *v37;
      if ( v32 == *v37 )
        goto LABEL_36;
    }
    if ( !v54 )
      v54 = v37;
    ++v94;
    v55 = v96 + 1;
    if ( 4 * ((int)v96 + 1) >= 3 * v97 )
      goto LABEL_135;
    if ( v97 - HIDWORD(v96) - v55 <= v97 >> 3 )
    {
      v83 = v35;
      v89 = ((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4);
      v85 = v32;
      sub_FAA9D0((__int64)&v94, v97);
      if ( v97 )
      {
        v80 = 1;
        v32 = v85;
        v55 = v96 + 1;
        v79 = 0;
        LODWORD(v35) = v83;
        a2 = (v97 - 1) & v89;
        v54 = &v95[5 * a2];
        v81 = *v54;
        if ( v85 == *v54 )
          goto LABEL_78;
        while ( v81 != -4096 )
        {
          if ( !v79 && v81 == -8192 )
            v79 = v54;
          a2 = (v97 - 1) & ((_DWORD)a2 + v80);
          v54 = &v95[5 * a2];
          v81 = *v54;
          if ( v85 == *v54 )
            goto LABEL_78;
          ++v80;
        }
        goto LABEL_147;
      }
      goto LABEL_164;
    }
LABEL_78:
    LODWORD(v96) = v55;
    if ( *v54 != -4096 )
      --HIDWORD(v96);
    *v54 = v32;
    v56 = v54 + 3;
    v40 = v54 + 1;
    *v40 = v56;
    v39 = 0;
    v40[1] = 0x400000000LL;
LABEL_38:
    *(_DWORD *)(*v40 + 4 * v39) = v35;
    ++*((_DWORD *)v40 + 2);
LABEL_39:
    ++v92;
    v6 += 2;
  }
  while ( v86 != v92 );
  v3 = v95;
  v41 = v97;
  v2 = 5LL * v97;
  if ( !(_DWORD)v96 )
    goto LABEL_41;
  v57 = &v95[v2];
  if ( &v95[v2] == v95 )
    goto LABEL_41;
  v58 = v95;
  while ( 1 )
  {
    a2 = *v58;
    v59 = v58;
    if ( *v58 != -8192 && a2 != -4096 )
      break;
    v58 += 5;
    if ( v57 == v58 )
      goto LABEL_41;
  }
  if ( v58 != v57 )
  {
    v60 = v90;
    while ( 1 )
    {
      v61 = *((unsigned int *)v59 + 4);
      if ( v61 > 1 )
        goto LABEL_98;
      v62 = **(_QWORD **)(a1 - 8);
      v63 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
      if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      {
        v64 = *(char **)(a2 - 8);
        v65 = v63 >> 5;
        v66 = &v64[v63];
        v67 = v63 >> 7;
        if ( !v67 )
          goto LABEL_120;
      }
      else
      {
        v66 = (char *)a2;
        v64 = (char *)(a2 - v63);
        v65 = v63 >> 5;
        v67 = v63 >> 7;
        if ( !v67 )
        {
LABEL_120:
          if ( v65 == 2 )
            goto LABEL_129;
          goto LABEL_121;
        }
      }
      v68 = &v64[128 * v67];
      do
      {
        if ( v62 == *(_QWORD *)v64 )
          goto LABEL_97;
        if ( v62 == *((_QWORD *)v64 + 4) )
        {
          v64 += 32;
          goto LABEL_97;
        }
        if ( v62 == *((_QWORD *)v64 + 8) )
        {
          v64 += 64;
          goto LABEL_97;
        }
        if ( v62 == *((_QWORD *)v64 + 12) )
        {
          v64 += 96;
          goto LABEL_97;
        }
        v64 += 128;
      }
      while ( v68 != v64 );
      v65 = (v66 - v64) >> 5;
      if ( v65 == 2 )
        goto LABEL_129;
LABEL_121:
      if ( v65 != 3 )
      {
        if ( v65 == 1 )
          goto LABEL_123;
        goto LABEL_112;
      }
      if ( v62 == *(_QWORD *)v64 )
        break;
      v64 += 32;
LABEL_129:
      if ( v62 == *(_QWORD *)v64 )
        break;
      v64 += 32;
LABEL_123:
      if ( v62 == *(_QWORD *)v64 )
        break;
LABEL_112:
      v59 += 5;
      if ( v59 == v57 )
        goto LABEL_116;
      while ( *v59 == -4096 || *v59 == -8192 )
      {
        v59 += 5;
        if ( v57 == v59 )
          goto LABEL_116;
      }
      if ( v57 == v59 )
      {
LABEL_116:
        v3 = v95;
        v90 = v60;
        v41 = v97;
        v2 = 5LL * v97;
        goto LABEL_41;
      }
      a2 = *v59;
    }
LABEL_97:
    if ( v64 == v66 )
      goto LABEL_112;
LABEL_98:
    v69 = (unsigned int *)v59[1];
    v70 = &v69[v61];
    if ( v70 == v69 )
    {
LABEL_111:
      v60 = 1;
      goto LABEL_112;
    }
    while ( 2 )
    {
      while ( 1 )
      {
        v73 = **(_QWORD **)(a1 - 8);
        v74 = *(_QWORD *)(a2 - 8) + 32LL * *v69;
        v75 = *(_QWORD *)v74;
        if ( v73 )
          break;
        if ( !v75 )
          goto LABEL_106;
        v76 = *(_QWORD *)(v74 + 8);
        **(_QWORD **)(v74 + 16) = v76;
        if ( !v76 )
        {
          *(_QWORD *)v74 = 0;
          goto LABEL_106;
        }
        ++v69;
        *(_QWORD *)(v76 + 16) = *(_QWORD *)(v74 + 16);
        *(_QWORD *)v74 = 0;
        if ( v70 == v69 )
          goto LABEL_111;
      }
      if ( v75 )
      {
        v71 = *(_QWORD *)(v74 + 8);
        **(_QWORD **)(v74 + 16) = v71;
        if ( v71 )
          *(_QWORD *)(v71 + 16) = *(_QWORD *)(v74 + 16);
      }
      *(_QWORD *)v74 = v73;
      v72 = *(_QWORD *)(v73 + 16);
      *(_QWORD *)(v74 + 8) = v72;
      if ( v72 )
        *(_QWORD *)(v72 + 16) = v74 + 8;
      *(_QWORD *)(v74 + 16) = v73 + 16;
      *(_QWORD *)(v73 + 16) = v74;
LABEL_106:
      if ( v70 == ++v69 )
        goto LABEL_111;
      continue;
    }
  }
LABEL_41:
  if ( v41 )
  {
    v42 = &v3[v2];
    do
    {
      if ( *v3 != -4096 && *v3 != -8192 )
      {
        v43 = (_QWORD *)v3[1];
        if ( v43 != v3 + 3 )
          _libc_free(v43, a2);
      }
      v3 += 5;
    }
    while ( v42 != v3 );
    v3 = v95;
    v2 = 5LL * v97;
  }
LABEL_49:
  sub_C7D6A0((__int64)v3, v2 * 8, 8);
  return v90;
}
