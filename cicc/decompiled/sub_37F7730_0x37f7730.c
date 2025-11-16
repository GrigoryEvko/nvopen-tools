// Function: sub_37F7730
// Address: 0x37f7730
//
_DWORD *__fastcall sub_37F7730(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v4; // r12
  __int64 v5; // r13
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  int v8; // edx
  __int16 *v9; // r15
  unsigned int v10; // r12d
  char v11; // al
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r8
  __int64 v15; // r13
  unsigned int v16; // eax
  _DWORD *v17; // rcx
  __int64 v18; // rax
  __int64 v19; // r14
  __int64 v20; // rcx
  __int64 *v21; // r9
  unsigned __int64 v22; // rbx
  __int64 v23; // rax
  unsigned __int64 v24; // rdi
  __int64 v25; // rax
  int v26; // r14d
  unsigned int v27; // esi
  __int64 v28; // rdi
  __int64 v29; // r8
  __int64 v30; // r11
  __int64 v31; // r9
  unsigned int j; // eax
  __int64 v33; // r12
  int v34; // edx
  int v35; // eax
  unsigned int v36; // esi
  int v37; // r14d
  __int64 v38; // rdi
  int v39; // r15d
  __int64 v40; // rcx
  __int64 *v41; // r9
  unsigned int v42; // edx
  _QWORD *v43; // rax
  __int64 v44; // r11
  _DWORD *result; // rax
  __int64 v46; // rax
  __int64 v47; // r14
  int v48; // r15d
  int v49; // eax
  int v50; // edx
  int v51; // eax
  int v52; // edx
  int v53; // eax
  int v54; // r8d
  __int64 v55; // r10
  unsigned int v56; // ecx
  __int64 v57; // rdi
  int v58; // eax
  __int64 *v59; // rsi
  int v60; // eax
  int v61; // esi
  __int64 v62; // rdi
  __int64 *v63; // r8
  unsigned int v64; // r12d
  int v65; // eax
  __int64 v66; // rcx
  int v67; // ecx
  __int64 v68; // rdx
  int v69; // esi
  __int64 v70; // rcx
  int v71; // edi
  unsigned int i; // eax
  int v73; // r8d
  unsigned int v74; // eax
  int v75; // eax
  int v76; // edx
  __int64 v77; // rsi
  int v78; // ecx
  __int64 v79; // rax
  unsigned int k; // r15d
  int v81; // edi
  unsigned int v82; // r15d
  unsigned int v83; // [rsp+14h] [rbp-5Ch]
  __int64 v84; // [rsp+18h] [rbp-58h]
  __int64 *v85; // [rsp+20h] [rbp-50h]
  __int64 v86; // [rsp+20h] [rbp-50h]
  __int64 v88; // [rsp+38h] [rbp-38h]

  v2 = a2;
  v4 = *(_QWORD *)(a2 + 32);
  v83 = *(_DWORD *)(*(_QWORD *)(a2 + 24) + 24LL);
  v88 = v4 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF);
  if ( v88 != v4 )
  {
    v5 = *(_QWORD *)(a2 + 32);
    while ( 1 )
    {
      v11 = *(_BYTE *)v5;
      if ( *(_BYTE *)v5 != 5 )
        goto LABEL_10;
      v26 = *(_DWORD *)(v5 + 24);
      if ( sub_37F43B0(a2, v26, *(__int64 **)(a1 + 216)) )
      {
        v27 = *(_DWORD *)(a1 + 632);
        v28 = a1 + 608;
        if ( !v27 )
        {
          ++*(_QWORD *)(a1 + 608);
LABEL_81:
          sub_37F7380(v28, 2 * v27);
          v67 = *(_DWORD *)(a1 + 632);
          if ( v67 )
          {
            v69 = v67 - 1;
            v70 = 0;
            v71 = 1;
            for ( i = v69
                    & (((0xBF58476D1CE4E5B9LL * (((unsigned __int64)(37 * v83) << 32) | (unsigned int)(37 * v26))) >> 31)
                     ^ (756364221 * v26)); ; i = v69 & v74 )
            {
              v68 = *(_QWORD *)(a1 + 616);
              v33 = v68 + 72LL * i;
              v73 = *(_DWORD *)v33;
              if ( v83 == *(_DWORD *)v33 && v26 == *(_DWORD *)(v33 + 4) )
                break;
              if ( v73 == -1 )
              {
                if ( *(_DWORD *)(v33 + 4) == 0x7FFFFFFF )
                {
                  if ( v70 )
                    v33 = v70;
                  v52 = *(_DWORD *)(a1 + 624) + 1;
                  goto LABEL_61;
                }
              }
              else if ( v73 == -2 && *(_DWORD *)(v33 + 4) == 0x80000000 && !v70 )
              {
                v70 = v68 + 72LL * i;
              }
              v74 = v71 + i;
              ++v71;
            }
            goto LABEL_100;
          }
LABEL_121:
          ++*(_DWORD *)(a1 + 624);
          BUG();
        }
        v29 = 1;
        v30 = 0;
        v31 = *(_QWORD *)(a1 + 616);
        for ( j = (((0xBF58476D1CE4E5B9LL * (((unsigned __int64)(37 * v83) << 32) | (unsigned int)(37 * v26))) >> 31)
                 ^ (756364221 * v26))
                & (v27 - 1); ; j = (v27 - 1) & v35 )
        {
          v33 = v31 + 72LL * j;
          v34 = *(_DWORD *)v33;
          if ( v83 == *(_DWORD *)v33 && v26 == *(_DWORD *)(v33 + 4) )
          {
            v46 = *(unsigned int *)(v33 + 16);
            v47 = v33 + 8;
            v48 = *(_DWORD *)(a1 + 456);
            if ( v46 + 1 > (unsigned __int64)*(unsigned int *)(v33 + 20) )
            {
              sub_C8D5F0(v33 + 8, (const void *)(v33 + 24), v46 + 1, 4u, v29, v31);
              v46 = *(unsigned int *)(v33 + 16);
            }
            goto LABEL_42;
          }
          if ( v34 == -1 )
            break;
          if ( v34 == -2 && *(_DWORD *)(v33 + 4) == 0x80000000 && !v30 )
            v30 = v31 + 72LL * j;
LABEL_33:
          v35 = v29 + j;
          v29 = (unsigned int)(v29 + 1);
        }
        if ( *(_DWORD *)(v33 + 4) != 0x7FFFFFFF )
          goto LABEL_33;
        v51 = *(_DWORD *)(a1 + 624);
        if ( v30 )
          v33 = v30;
        ++*(_QWORD *)(a1 + 608);
        v52 = v51 + 1;
        if ( 4 * (v51 + 1) >= 3 * v27 )
          goto LABEL_81;
        if ( v27 - *(_DWORD *)(a1 + 628) - v52 <= v27 >> 3 )
        {
          sub_37F7380(v28, v27);
          v75 = *(_DWORD *)(a1 + 632);
          if ( v75 )
          {
            v76 = v75 - 1;
            v78 = 1;
            v79 = 0;
            for ( k = v76
                    & (((0xBF58476D1CE4E5B9LL * (((unsigned __int64)(37 * v83) << 32) | (unsigned int)(37 * v26))) >> 31)
                     ^ (756364221 * v26)); ; k = v76 & v82 )
            {
              v77 = *(_QWORD *)(a1 + 616);
              v33 = v77 + 72LL * k;
              v81 = *(_DWORD *)v33;
              if ( v83 == *(_DWORD *)v33 && v26 == *(_DWORD *)(v33 + 4) )
                break;
              if ( v81 == -1 )
              {
                if ( *(_DWORD *)(v33 + 4) == 0x7FFFFFFF )
                {
                  if ( v79 )
                    v33 = v79;
                  v52 = *(_DWORD *)(a1 + 624) + 1;
                  goto LABEL_61;
                }
              }
              else if ( v81 == -2 && *(_DWORD *)(v33 + 4) == 0x80000000 && !v79 )
              {
                v79 = v77 + 72LL * k;
              }
              v82 = v78 + k;
              ++v78;
            }
LABEL_100:
            v52 = *(_DWORD *)(a1 + 624) + 1;
            goto LABEL_61;
          }
          goto LABEL_121;
        }
LABEL_61:
        *(_DWORD *)(a1 + 624) = v52;
        if ( *(_DWORD *)v33 != -1 || *(_DWORD *)(v33 + 4) != 0x7FFFFFFF )
          --*(_DWORD *)(a1 + 628);
        *(_DWORD *)(v33 + 4) = v26;
        v47 = v33 + 8;
        *(_DWORD *)v33 = v83;
        *(_QWORD *)(v33 + 8) = v33 + 24;
        *(_QWORD *)(v33 + 16) = 0xC00000000LL;
        v46 = 0;
        v48 = *(_DWORD *)(a1 + 456);
LABEL_42:
        *(_DWORD *)(*(_QWORD *)v47 + 4 * v46) = v48;
        ++*(_DWORD *)(v47 + 8);
        v11 = *(_BYTE *)v5;
LABEL_10:
        if ( !v11 )
        {
          v12 = *(unsigned int *)(v5 + 8);
          if ( (_DWORD)v12 )
          {
            if ( (*(_BYTE *)(v5 + 3) & 0x10) != 0 )
              break;
          }
        }
      }
LABEL_8:
      v5 += 40;
      if ( v88 == v5 )
      {
        v2 = a2;
        goto LABEL_35;
      }
    }
    v13 = *(_QWORD *)(a1 + 208);
    v14 = v5;
    v15 = a1;
    v16 = *(_DWORD *)(*(_QWORD *)(v13 + 8) + 24 * v12 + 16);
    v10 = v16 & 0xFFF;
    v9 = (__int16 *)(*(_QWORD *)(v13 + 56) + 2LL * (v16 >> 12));
    while ( 1 )
    {
      if ( !v9 )
      {
LABEL_7:
        a1 = v15;
        v5 = v14;
        goto LABEL_8;
      }
      v17 = (_DWORD *)(*(_QWORD *)(v15 + 320) + 4LL * v10);
      v18 = *(int *)(v15 + 456);
      if ( *v17 != (_DWORD)v18 )
        break;
LABEL_6:
      v8 = *v9++;
      v10 += v8;
      if ( !(_WORD)v8 )
        goto LABEL_7;
    }
    *v17 = v18;
    v19 = 4 * v18 + 2;
    v20 = *(_QWORD *)(*(_QWORD *)(v15 + 496) + 24LL * v83);
    v21 = (__int64 *)(v20 + 8LL * v10);
    v22 = *v21 & 0xFFFFFFFFFFFFFFFELL;
    if ( !v22 )
    {
      *v21 = v19;
      goto LABEL_6;
    }
    if ( (*v21 & 1) != 0 )
    {
      v6 = *(unsigned int *)(v22 + 8);
      v7 = v6 + 1;
      if ( v6 + 1 <= (unsigned __int64)*(unsigned int *)(v22 + 12) )
      {
LABEL_5:
        *(_QWORD *)(*(_QWORD *)v22 + 8 * v6) = v19;
        ++*(_DWORD *)(v22 + 8);
        goto LABEL_6;
      }
    }
    else
    {
      v84 = v14;
      v85 = (__int64 *)(v20 + 8LL * v10);
      v23 = sub_22077B0(0x30u);
      v21 = v85;
      v14 = v84;
      if ( v23 )
      {
        *(_QWORD *)v23 = v23 + 16;
        *(_QWORD *)(v23 + 8) = 0x400000000LL;
      }
      v24 = v23 & 0xFFFFFFFFFFFFFFFELL;
      *v85 = v23 | 1;
      v25 = *(unsigned int *)((v23 & 0xFFFFFFFFFFFFFFFELL) + 8);
      if ( v25 + 1 > (unsigned __int64)*(unsigned int *)(v24 + 12) )
      {
        sub_C8D5F0(v24, (const void *)(v24 + 16), v25 + 1, 8u, v84, (__int64)v85);
        v14 = v84;
        v21 = v85;
        v25 = *(unsigned int *)(v24 + 8);
      }
      *(_QWORD *)(*(_QWORD *)v24 + 8 * v25) = v22;
      ++*(_DWORD *)(v24 + 8);
      v22 = *v21 & 0xFFFFFFFFFFFFFFFELL;
      v6 = *(unsigned int *)(v22 + 8);
      v7 = v6 + 1;
      if ( v6 + 1 <= (unsigned __int64)*(unsigned int *)(v22 + 12) )
        goto LABEL_5;
    }
    v86 = v14;
    sub_C8D5F0(v22, (const void *)(v22 + 16), v7, 8u, v14, (__int64)v21);
    v6 = *(unsigned int *)(v22 + 8);
    v14 = v86;
    goto LABEL_5;
  }
LABEL_35:
  v36 = *(_DWORD *)(a1 + 488);
  v37 = *(_DWORD *)(a1 + 456);
  v38 = a1 + 464;
  if ( !v36 )
  {
    ++*(_QWORD *)(a1 + 464);
    goto LABEL_67;
  }
  v39 = 1;
  v40 = *(_QWORD *)(a1 + 472);
  v41 = 0;
  v42 = (v36 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
  v43 = (_QWORD *)(v40 + 16LL * v42);
  v44 = *v43;
  if ( *v43 == v2 )
  {
LABEL_37:
    result = v43 + 1;
    goto LABEL_38;
  }
  while ( v44 != -4096 )
  {
    if ( !v41 && v44 == -8192 )
      v41 = v43;
    v42 = (v36 - 1) & (v39 + v42);
    v43 = (_QWORD *)(v40 + 16LL * v42);
    v44 = *v43;
    if ( *v43 == v2 )
      goto LABEL_37;
    ++v39;
  }
  if ( !v41 )
    v41 = v43;
  v49 = *(_DWORD *)(a1 + 480);
  ++*(_QWORD *)(a1 + 464);
  v50 = v49 + 1;
  if ( 4 * (v49 + 1) >= 3 * v36 )
  {
LABEL_67:
    sub_354C5D0(v38, 2 * v36);
    v53 = *(_DWORD *)(a1 + 488);
    if ( v53 )
    {
      v54 = v53 - 1;
      v55 = *(_QWORD *)(a1 + 472);
      v56 = (v53 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
      v41 = (__int64 *)(v55 + 16LL * v56);
      v57 = *v41;
      v50 = *(_DWORD *)(a1 + 480) + 1;
      if ( *v41 != v2 )
      {
        v58 = 1;
        v59 = 0;
        while ( v57 != -4096 )
        {
          if ( v57 == -8192 && !v59 )
            v59 = v41;
          v56 = v54 & (v58 + v56);
          v41 = (__int64 *)(v55 + 16LL * v56);
          v57 = *v41;
          if ( *v41 == v2 )
            goto LABEL_53;
          ++v58;
        }
        if ( v59 )
          v41 = v59;
      }
      goto LABEL_53;
    }
    goto LABEL_122;
  }
  if ( v36 - *(_DWORD *)(a1 + 484) - v50 <= v36 >> 3 )
  {
    sub_354C5D0(v38, v36);
    v60 = *(_DWORD *)(a1 + 488);
    if ( v60 )
    {
      v61 = v60 - 1;
      v62 = *(_QWORD *)(a1 + 472);
      v63 = 0;
      v64 = (v60 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
      v50 = *(_DWORD *)(a1 + 480) + 1;
      v65 = 1;
      v41 = (__int64 *)(v62 + 16LL * v64);
      v66 = *v41;
      if ( *v41 != v2 )
      {
        while ( v66 != -4096 )
        {
          if ( !v63 && v66 == -8192 )
            v63 = v41;
          v64 = v61 & (v65 + v64);
          v41 = (__int64 *)(v62 + 16LL * v64);
          v66 = *v41;
          if ( *v41 == v2 )
            goto LABEL_53;
          ++v65;
        }
        if ( v63 )
          v41 = v63;
      }
      goto LABEL_53;
    }
LABEL_122:
    ++*(_DWORD *)(a1 + 480);
    BUG();
  }
LABEL_53:
  *(_DWORD *)(a1 + 480) = v50;
  if ( *v41 != -4096 )
    --*(_DWORD *)(a1 + 484);
  *v41 = v2;
  result = v41 + 1;
  *((_DWORD *)v41 + 2) = 0;
LABEL_38:
  *result = v37;
  ++*(_DWORD *)(a1 + 456);
  return result;
}
