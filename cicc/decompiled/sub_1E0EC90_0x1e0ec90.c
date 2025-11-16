// Function: sub_1E0EC90
// Address: 0x1e0ec90
//
unsigned __int64 __fastcall sub_1E0EC90(
        unsigned __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int64 result; // rax
  __int64 v7; // rcx
  __int64 v9; // rdx
  __int64 v10; // rbx
  __int64 v11; // r12
  unsigned int v12; // esi
  __int64 *v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rsi
  unsigned int v16; // r13d
  unsigned int v17; // esi
  __int64 v18; // r10
  __int64 *v19; // rax
  __int64 v20; // rdi
  unsigned int v21; // esi
  unsigned int v22; // edi
  __int64 *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r14
  __int64 v26; // r12
  __int64 v27; // r15
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rdi
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rsi
  __int64 v35; // rdi
  int v36; // eax
  __int64 v37; // rdx
  char *v38; // rcx
  char *v39; // rax
  __int64 v40; // rax
  int v41; // eax
  int v42; // eax
  int v43; // esi
  int v44; // esi
  unsigned int v45; // edx
  __int64 v46; // rdi
  int v47; // r15d
  __int64 v48; // r10
  int v49; // esi
  int v50; // esi
  int v51; // r15d
  unsigned int v52; // edx
  __int64 v53; // rdi
  int v54; // r11d
  int v55; // eax
  int v56; // eax
  int v57; // esi
  int v58; // esi
  unsigned int v59; // edx
  __int64 v60; // rdi
  int v61; // r12d
  __int64 v62; // r10
  int v63; // edx
  int v64; // edx
  __int64 v65; // rdi
  int v66; // r11d
  unsigned int v67; // r12d
  __int64 v68; // rsi
  int v69; // r12d
  int v70; // r12d
  __int64 v71; // r11
  int v72; // edx
  int v73; // r12d
  __int64 *v74; // r10
  int v75; // ecx
  int v76; // r12d
  int v77; // r12d
  __int64 v78; // r11
  __int64 *v79; // rsi
  int v80; // edi
  int v81; // edi
  unsigned int v82; // [rsp+8h] [rbp-68h]
  __int64 v83; // [rsp+10h] [rbp-60h]
  unsigned int v84; // [rsp+1Ch] [rbp-54h]
  __int64 v85; // [rsp+20h] [rbp-50h]
  __int64 v87; // [rsp+30h] [rbp-40h]
  __int64 v88; // [rsp+30h] [rbp-40h]
  __int64 v89; // [rsp+30h] [rbp-40h]
  __int64 v90; // [rsp+30h] [rbp-40h]
  int v91; // [rsp+30h] [rbp-40h]
  __int64 v92; // [rsp+30h] [rbp-40h]
  __int64 v93; // [rsp+30h] [rbp-40h]
  __int64 v94; // [rsp+30h] [rbp-40h]
  __int64 v95; // [rsp+30h] [rbp-40h]
  int v96; // [rsp+38h] [rbp-38h]

  result = a1;
  v7 = 0;
  v9 = *(_QWORD *)(a1 + 408);
  v84 = 0;
  v83 = a1 + 408;
  if ( v9 == *(_QWORD *)(a1 + 416) )
    return result;
  do
  {
    v10 = v9 + 120 * v7;
    v85 = 120 * v7;
    v11 = *(_QWORD *)(v10 + 88);
    if ( !v11 )
      goto LABEL_10;
    if ( (*(_QWORD *)v11 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    {
      if ( (*(_BYTE *)(v11 + 9) & 0xC) == 8 )
      {
        *(_BYTE *)(v11 + 8) |= 4u;
        v9 = sub_38CE440(*(_QWORD *)(v11 + 24));
        *(_QWORD *)v11 = v9 | *(_QWORD *)v11 & 7LL;
        if ( v9 )
        {
          v9 = *(_QWORD *)(v10 + 88);
LABEL_14:
          if ( v9 )
          {
LABEL_15:
            v9 = *(_QWORD *)(a1 + 408);
            goto LABEL_16;
          }
LABEL_10:
          if ( *(_QWORD *)v10 )
          {
            sub_1E0BEF0(v83, *(_QWORD *)(a1 + 408) + v85, v9, v7, a5, a6);
            v9 = *(_QWORD *)(a1 + 408);
            goto LABEL_42;
          }
          goto LABEL_15;
        }
      }
      if ( !a2 )
      {
LABEL_9:
        *(_QWORD *)(v10 + 88) = 0;
        goto LABEL_10;
      }
      v12 = *(_DWORD *)(a2 + 24);
      if ( v12 )
      {
        v9 = *(_QWORD *)(v10 + 88);
        LODWORD(a6) = v12 - 1;
        a5 = *(_QWORD *)(a2 + 8);
        v7 = (v12 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v13 = (__int64 *)(a5 + 16 * v7);
        v14 = *v13;
        if ( v9 == *v13 )
        {
LABEL_8:
          if ( v13[1] )
            goto LABEL_14;
          goto LABEL_9;
        }
        v73 = 1;
        v74 = 0;
        while ( v14 != -8 )
        {
          if ( !v74 && v14 == -16 )
            v74 = v13;
          v7 = (unsigned int)a6 & (v73 + (_DWORD)v7);
          v13 = (__int64 *)(a5 + 16LL * (unsigned int)v7);
          v14 = *v13;
          if ( v9 == *v13 )
            goto LABEL_8;
          ++v73;
        }
        v75 = *(_DWORD *)(a2 + 16);
        if ( v74 )
          v13 = v74;
        ++*(_QWORD *)a2;
        v72 = v75 + 1;
        if ( 4 * (v75 + 1) < 3 * v12 )
        {
          v7 = v12 - *(_DWORD *)(a2 + 20) - v72;
          if ( (unsigned int)v7 > v12 >> 3 )
            goto LABEL_106;
          sub_1E0EAD0(a2, v12);
          v76 = *(_DWORD *)(a2 + 24);
          if ( !v76 )
          {
LABEL_159:
            ++*(_DWORD *)(a2 + 16);
            BUG();
          }
          a6 = *(_QWORD *)(v10 + 88);
          v77 = v76 - 1;
          v78 = *(_QWORD *)(a2 + 8);
          v79 = 0;
          v72 = *(_DWORD *)(a2 + 16) + 1;
          v80 = 1;
          v7 = v77 & (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4));
          v13 = (__int64 *)(v78 + 16 * v7);
          a5 = *v13;
          if ( a6 == *v13 )
            goto LABEL_106;
          while ( a5 != -8 )
          {
            if ( a5 == -16 && !v79 )
              v79 = v13;
            v7 = v77 & (unsigned int)(v80 + v7);
            v13 = (__int64 *)(v78 + 16LL * (unsigned int)v7);
            a5 = *v13;
            if ( a6 == *v13 )
              goto LABEL_106;
            ++v80;
          }
          goto LABEL_123;
        }
      }
      else
      {
        ++*(_QWORD *)a2;
      }
      sub_1E0EAD0(a2, 2 * v12);
      v69 = *(_DWORD *)(a2 + 24);
      if ( !v69 )
        goto LABEL_159;
      a6 = *(_QWORD *)(v10 + 88);
      v70 = v69 - 1;
      v71 = *(_QWORD *)(a2 + 8);
      v72 = *(_DWORD *)(a2 + 16) + 1;
      v7 = v70 & (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4));
      v13 = (__int64 *)(v71 + 16 * v7);
      a5 = *v13;
      if ( a6 == *v13 )
        goto LABEL_106;
      v81 = 1;
      v79 = 0;
      while ( a5 != -8 )
      {
        if ( !v79 && a5 == -16 )
          v79 = v13;
        v7 = v70 & (unsigned int)(v81 + v7);
        v13 = (__int64 *)(v71 + 16LL * (unsigned int)v7);
        a5 = *v13;
        if ( a6 == *v13 )
          goto LABEL_106;
        ++v81;
      }
LABEL_123:
      if ( v79 )
        v13 = v79;
LABEL_106:
      *(_DWORD *)(a2 + 16) = v72;
      if ( *v13 != -8 )
        --*(_DWORD *)(a2 + 20);
      v9 = *(_QWORD *)(v10 + 88);
      v13[1] = 0;
      *v13 = v9;
      *(_QWORD *)(v10 + 88) = 0;
      goto LABEL_10;
    }
LABEL_16:
    v15 = v9 + v85;
    v96 = *(_DWORD *)(v9 + v85 + 16);
    if ( !v96 )
      goto LABEL_49;
    a5 = a2;
    v16 = 0;
    do
    {
      while ( 1 )
      {
        v25 = 8LL * v16;
        v26 = *(_QWORD *)(*(_QWORD *)(v10 + 8) + v25);
        v27 = *(_QWORD *)(*(_QWORD *)(v10 + 32) + v25);
        if ( (*(_QWORD *)v26 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          goto LABEL_22;
        if ( (*(_BYTE *)(v26 + 9) & 0xC) == 8 )
        {
          *(_BYTE *)(v26 + 8) |= 4u;
          v87 = a5;
          v28 = sub_38CE440(*(_QWORD *)(v26 + 24));
          a5 = v87;
          *(_QWORD *)v26 = v28 | *(_QWORD *)v26 & 7LL;
          if ( v28 )
            goto LABEL_22;
          if ( !v87 )
            goto LABEL_33;
        }
        else if ( !a5 )
        {
          goto LABEL_33;
        }
        v17 = *(_DWORD *)(a5 + 24);
        if ( !v17 )
        {
          ++*(_QWORD *)a5;
          goto LABEL_60;
        }
        v18 = *(_QWORD *)(a5 + 8);
        LODWORD(a6) = (v17 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v19 = (__int64 *)(v18 + 16LL * (unsigned int)a6);
        v20 = *v19;
        if ( v26 != *v19 )
          break;
LABEL_21:
        if ( !v19[1] )
          goto LABEL_33;
LABEL_22:
        if ( (*(_QWORD *)v27 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          goto LABEL_28;
        if ( (*(_BYTE *)(v27 + 9) & 0xC) == 8 )
        {
          *(_BYTE *)(v27 + 8) |= 4u;
          v90 = a5;
          v40 = sub_38CE440(*(_QWORD *)(v27 + 24));
          a5 = v90;
          *(_QWORD *)v27 = v40 | *(_QWORD *)v27 & 7LL;
          if ( v40 )
            goto LABEL_28;
        }
        if ( !a5 )
          goto LABEL_33;
        v21 = *(_DWORD *)(a5 + 24);
        if ( !v21 )
        {
          ++*(_QWORD *)a5;
          goto LABEL_85;
        }
        a6 = *(_QWORD *)(a5 + 8);
        v22 = (v21 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
        v23 = (__int64 *)(a6 + 16LL * v22);
        v24 = *v23;
        if ( v27 != *v23 )
        {
          v54 = 1;
          v7 = 0;
          while ( v24 != -8 )
          {
            if ( !v7 && v24 == -16 )
              v7 = (__int64)v23;
            v22 = (v21 - 1) & (v54 + v22);
            v23 = (__int64 *)(a6 + 16LL * v22);
            v24 = *v23;
            if ( v27 == *v23 )
              goto LABEL_27;
            ++v54;
          }
          if ( !v7 )
            v7 = (__int64)v23;
          v55 = *(_DWORD *)(a5 + 16);
          ++*(_QWORD *)a5;
          v56 = v55 + 1;
          if ( 4 * v56 >= 3 * v21 )
          {
LABEL_85:
            v94 = a5;
            sub_1E0EAD0(a5, 2 * v21);
            a5 = v94;
            v57 = *(_DWORD *)(v94 + 24);
            if ( !v57 )
              goto LABEL_158;
            v58 = v57 - 1;
            a6 = *(_QWORD *)(v94 + 8);
            v59 = v58 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
            v7 = a6 + 16LL * v59;
            v60 = *(_QWORD *)v7;
            v56 = *(_DWORD *)(v94 + 16) + 1;
            if ( v27 != *(_QWORD *)v7 )
            {
              v61 = 1;
              v62 = 0;
              while ( v60 != -8 )
              {
                if ( !v62 && v60 == -16 )
                  v62 = v7;
                v59 = v58 & (v61 + v59);
                v7 = a6 + 16LL * v59;
                v60 = *(_QWORD *)v7;
                if ( v27 == *(_QWORD *)v7 )
                  goto LABEL_81;
                ++v61;
              }
              if ( v62 )
                v7 = v62;
            }
          }
          else if ( v21 - *(_DWORD *)(a5 + 20) - v56 <= v21 >> 3 )
          {
            v95 = a5;
            sub_1E0EAD0(a5, v21);
            a5 = v95;
            v63 = *(_DWORD *)(v95 + 24);
            if ( !v63 )
            {
LABEL_158:
              ++*(_DWORD *)(a5 + 16);
              BUG();
            }
            v64 = v63 - 1;
            v65 = *(_QWORD *)(v95 + 8);
            v66 = 1;
            a6 = 0;
            v67 = v64 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
            v7 = v65 + 16LL * v67;
            v68 = *(_QWORD *)v7;
            v56 = *(_DWORD *)(v95 + 16) + 1;
            if ( v27 != *(_QWORD *)v7 )
            {
              while ( v68 != -8 )
              {
                if ( v68 == -16 && !a6 )
                  a6 = v7;
                v67 = v64 & (v66 + v67);
                v7 = v65 + 16LL * v67;
                v68 = *(_QWORD *)v7;
                if ( v27 == *(_QWORD *)v7 )
                  goto LABEL_81;
                ++v66;
              }
              if ( a6 )
                v7 = a6;
            }
          }
LABEL_81:
          *(_DWORD *)(a5 + 16) = v56;
          if ( *(_QWORD *)v7 != -8 )
            --*(_DWORD *)(a5 + 20);
          *(_QWORD *)v7 = v27;
          *(_QWORD *)(v7 + 8) = 0;
          goto LABEL_33;
        }
LABEL_27:
        if ( !v23[1] )
          goto LABEL_33;
LABEL_28:
        if ( v96 == ++v16 )
          goto LABEL_38;
      }
      v91 = 1;
      v7 = 0;
      while ( v20 != -8 )
      {
        if ( !v7 && v20 == -16 )
          v7 = (__int64)v19;
        LODWORD(a6) = (v17 - 1) & (v91 + a6);
        v19 = (__int64 *)(v18 + 16LL * (unsigned int)a6);
        v20 = *v19;
        if ( v26 == *v19 )
          goto LABEL_21;
        ++v91;
      }
      if ( !v7 )
        v7 = (__int64)v19;
      v41 = *(_DWORD *)(a5 + 16);
      ++*(_QWORD *)a5;
      v42 = v41 + 1;
      if ( 4 * v42 < 3 * v17 )
      {
        LODWORD(a6) = v17 >> 3;
        if ( v17 - *(_DWORD *)(a5 + 20) - v42 > v17 >> 3 )
          goto LABEL_56;
        v93 = a5;
        v82 = ((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4);
        sub_1E0EAD0(a5, v17);
        a5 = v93;
        v49 = *(_DWORD *)(v93 + 24);
        if ( !v49 )
        {
LABEL_160:
          ++*(_DWORD *)(a5 + 16);
          BUG();
        }
        v50 = v49 - 1;
        a6 = *(_QWORD *)(v93 + 8);
        v51 = 1;
        v48 = 0;
        v52 = v50 & v82;
        v7 = a6 + 16LL * (v50 & v82);
        v53 = *(_QWORD *)v7;
        v42 = *(_DWORD *)(v93 + 16) + 1;
        if ( v26 == *(_QWORD *)v7 )
          goto LABEL_56;
        while ( v53 != -8 )
        {
          if ( !v48 && v53 == -16 )
            v48 = v7;
          v52 = v50 & (v51 + v52);
          v7 = a6 + 16LL * v52;
          v53 = *(_QWORD *)v7;
          if ( v26 == *(_QWORD *)v7 )
            goto LABEL_56;
          ++v51;
        }
        goto LABEL_64;
      }
LABEL_60:
      v92 = a5;
      sub_1E0EAD0(a5, 2 * v17);
      a5 = v92;
      v43 = *(_DWORD *)(v92 + 24);
      if ( !v43 )
        goto LABEL_160;
      v44 = v43 - 1;
      a6 = *(_QWORD *)(v92 + 8);
      v45 = v44 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
      v7 = a6 + 16LL * v45;
      v46 = *(_QWORD *)v7;
      v42 = *(_DWORD *)(v92 + 16) + 1;
      if ( v26 == *(_QWORD *)v7 )
        goto LABEL_56;
      v47 = 1;
      v48 = 0;
      while ( v46 != -8 )
      {
        if ( !v48 && v46 == -16 )
          v48 = v7;
        v45 = v44 & (v47 + v45);
        v7 = a6 + 16LL * v45;
        v46 = *(_QWORD *)v7;
        if ( v26 == *(_QWORD *)v7 )
          goto LABEL_56;
        ++v47;
      }
LABEL_64:
      if ( v48 )
        v7 = v48;
LABEL_56:
      *(_DWORD *)(a5 + 16) = v42;
      if ( *(_QWORD *)v7 != -8 )
        --*(_DWORD *)(a5 + 20);
      *(_QWORD *)v7 = v26;
      *(_QWORD *)(v7 + 8) = 0;
LABEL_33:
      v29 = *(_QWORD *)(v10 + 8);
      v30 = v29 + v25;
      v31 = *(unsigned int *)(v10 + 16);
      v32 = v29 + 8 * v31;
      if ( v32 != v30 + 8 )
      {
        v88 = a5;
        memmove((void *)v30, (const void *)(v30 + 8), v32 - (v30 + 8));
        LODWORD(v31) = *(_DWORD *)(v10 + 16);
        a5 = v88;
      }
      v33 = *(_QWORD *)(v10 + 32);
      v34 = *(unsigned int *)(v10 + 40);
      *(_DWORD *)(v10 + 16) = v31 - 1;
      v35 = v33 + v25;
      v36 = v34;
      v37 = v33 + 8 * v34;
      if ( v37 != v35 + 8 )
      {
        v89 = a5;
        memmove((void *)v35, (const void *)(v35 + 8), v37 - (v35 + 8));
        v36 = *(_DWORD *)(v10 + 40);
        a5 = v89;
      }
      --v96;
      *(_DWORD *)(v10 + 40) = v36 - 1;
    }
    while ( v96 != v16 );
LABEL_38:
    a2 = a5;
    v9 = *(_QWORD *)(a1 + 408);
    v15 = v9 + v85;
    if ( !*(_DWORD *)(v9 + v85 + 16) )
    {
LABEL_49:
      sub_1E0BEF0(v83, v15, v9, v7, a5, a6);
      v9 = *(_QWORD *)(a1 + 408);
      goto LABEL_42;
    }
    v38 = *(char **)(v10 + 104);
    v39 = *(char **)(v10 + 96);
    if ( (!*(_QWORD *)v10 || v38 - v39 == 4 && !*(_DWORD *)v39) && v38 != v39 )
    {
      *(_QWORD *)(v10 + 104) = v39;
      v9 = *(_QWORD *)(a1 + 408);
    }
    ++v84;
LABEL_42:
    v7 = v84;
    result = 0xEEEEEEEEEEEEEEEFLL * ((*(_QWORD *)(a1 + 416) - v9) >> 3);
  }
  while ( v84 != result );
  return result;
}
