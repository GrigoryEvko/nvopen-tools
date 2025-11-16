// Function: sub_29E9750
// Address: 0x29e9750
//
void __fastcall sub_29E9750(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v5; // rbx
  __int64 *v6; // r12
  __int64 v8; // r9
  int v9; // r14d
  __int64 *v10; // rdi
  __int64 *v11; // rax
  __int64 v12; // rsi
  __int64 *v13; // r14
  __int64 v14; // r15
  __int64 v15; // r15
  __int64 *v16; // rdi
  __int64 *v17; // rax
  __int64 v18; // r8
  __int64 v19; // r9
  unsigned __int64 v20; // rdx
  __int64 *v21; // r14
  unsigned __int64 v22; // rsi
  int v23; // eax
  __int64 v24; // rsi
  __int64 **v25; // rcx
  __int64 **v26; // rdx
  unsigned int v27; // esi
  int v28; // ecx
  __int64 v29; // rdx
  int v30; // eax
  __int64 v31; // rsi
  __int64 **v32; // rax
  int v33; // eax
  int v34; // ecx
  int v35; // r10d
  __int64 v36; // rdx
  __int64 v37; // rsi
  __int64 v38; // r11
  __int64 *v39; // rdi
  __int64 v40; // rax
  __int64 v41; // r11
  __int64 *v42; // r12
  int v43; // r10d
  unsigned int v44; // edi
  __int64 *v45; // rax
  __int64 v46; // rdi
  __int64 **v47; // rbx
  __int64 **v48; // r12
  __int64 v49; // rdi
  __int64 *v50; // rax
  unsigned __int8 v51; // al
  _QWORD *v52; // rbx
  __int64 v53; // rdx
  _QWORD *v54; // r15
  int v55; // r10d
  __int64 *v56; // rdx
  unsigned int v57; // edi
  __int64 *v58; // rax
  __int64 v59; // rcx
  _BYTE *v60; // r12
  __int64 v61; // rax
  unsigned __int64 v62; // rdx
  unsigned int v63; // esi
  int v64; // eax
  int v65; // edi
  __int64 v66; // rsi
  unsigned int v67; // eax
  int v68; // ecx
  int v69; // eax
  int v70; // eax
  int v71; // eax
  __int64 v72; // rdi
  unsigned int v73; // r14d
  __int64 v74; // rsi
  int v75; // eax
  int v76; // eax
  int v77; // r9d
  __int64 v78; // rdi
  int v79; // ebx
  __int64 *v80; // r10
  int v81; // r8d
  unsigned int v82; // ebx
  int v83; // r10d
  int v84; // r11d
  __int64 *v85; // r10
  int v86; // r10d
  unsigned int v87; // r10d
  __int64 v88; // [rsp+10h] [rbp-120h]
  __int64 *v89; // [rsp+20h] [rbp-110h]
  unsigned int v90; // [rsp+28h] [rbp-108h]
  __int64 v91; // [rsp+28h] [rbp-108h]
  __int64 v92; // [rsp+28h] [rbp-108h]
  __int64 **v93; // [rsp+28h] [rbp-108h]
  __int64 *v94; // [rsp+38h] [rbp-F8h]
  __int64 *v95; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v96; // [rsp+48h] [rbp-E8h]
  _BYTE v97[32]; // [rsp+50h] [rbp-E0h] BYREF
  __int64 **v98; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v99; // [rsp+78h] [rbp-B8h]
  _BYTE v100[176]; // [rsp+80h] [rbp-B0h] BYREF

  v5 = *(__int64 **)(a1 + 32);
  v98 = (__int64 **)v100;
  v99 = 0x1000000000LL;
  v6 = &v5[*(unsigned int *)(a1 + 40)];
  if ( v5 == v6 )
    goto LABEL_49;
  v88 = a1 + 48;
  do
  {
    v15 = *v5;
    v16 = (__int64 *)(*(_QWORD *)(*v5 + 8) & 0xFFFFFFFFFFFFFFF8LL);
    if ( (*(_QWORD *)(*v5 + 8) & 4) != 0 )
      v16 = (__int64 *)*v16;
    v17 = (__int64 *)sub_B9C770(v16, 0, 0, 2, 1);
    v20 = (unsigned int)v99;
    v95 = v17;
    v21 = v17;
    v22 = (unsigned int)v99 + 1LL;
    v23 = v99;
    if ( v22 > HIDWORD(v99) )
    {
      if ( v98 > &v95 || (v20 = (unsigned __int64)&v98[(unsigned int)v99], (unsigned __int64)&v95 >= v20) )
      {
        sub_29E6920((__int64)&v98, v22, v20, HIDWORD(v99), v18, v19);
        v20 = (unsigned int)v99;
        v24 = (__int64)v98;
        v25 = &v95;
        v23 = v99;
      }
      else
      {
        v93 = v98;
        sub_29E6920((__int64)&v98, v22, v20, HIDWORD(v99), v18, v19);
        v24 = (__int64)v98;
        v20 = (unsigned int)v99;
        v25 = (__int64 **)((char *)v98 + (char *)&v95 - (char *)v93);
        v23 = v99;
      }
    }
    else
    {
      v24 = (__int64)v98;
      v25 = &v95;
    }
    v26 = (__int64 **)(v24 + 8 * v20);
    if ( v26 )
    {
      *v26 = *v25;
      *v25 = 0;
      v21 = v95;
      v23 = v99;
    }
    LODWORD(v99) = v23 + 1;
    if ( v21 )
      sub_BA65D0((__int64)v21, v24, (__int64)v26, (__int64)v25, v18);
    v27 = *(_DWORD *)(a1 + 72);
    if ( !v27 )
    {
      ++*(_QWORD *)(a1 + 48);
      goto LABEL_19;
    }
    v8 = *(_QWORD *)(a1 + 56);
    v9 = 1;
    v10 = 0;
    a5 = (v27 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
    v11 = (__int64 *)(v8 + 16 * a5);
    a4 = *v11;
    if ( v15 != *v11 )
    {
      while ( a4 != -4096 )
      {
        if ( a4 == -8192 && !v10 )
          v10 = v11;
        a5 = (v27 - 1) & (v9 + (_DWORD)a5);
        v11 = (__int64 *)(v8 + 16LL * (unsigned int)a5);
        a4 = *v11;
        if ( v15 == *v11 )
          goto LABEL_4;
        ++v9;
      }
      if ( !v10 )
        v10 = v11;
      v33 = *(_DWORD *)(a1 + 64);
      ++*(_QWORD *)(a1 + 48);
      v30 = v33 + 1;
      if ( 4 * v30 < 3 * v27 )
      {
        a4 = v27 - *(_DWORD *)(a1 + 68) - v30;
        a5 = v27 >> 3;
        if ( (unsigned int)a4 > (unsigned int)a5 )
          goto LABEL_21;
        v90 = ((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4);
        sub_29E9530(v88, v27);
        v34 = *(_DWORD *)(a1 + 72);
        if ( !v34 )
        {
LABEL_165:
          ++*(_DWORD *)(a1 + 64);
          BUG();
        }
        a4 = (unsigned int)(v34 - 1);
        a5 = *(_QWORD *)(a1 + 56);
        v8 = 0;
        v35 = 1;
        LODWORD(v36) = a4 & v90;
        v30 = *(_DWORD *)(a1 + 64) + 1;
        v10 = (__int64 *)(a5 + 16LL * ((unsigned int)a4 & v90));
        v37 = *v10;
        if ( v15 == *v10 )
          goto LABEL_21;
        while ( v37 != -4096 )
        {
          if ( !v8 && v37 == -8192 )
            v8 = (__int64)v10;
          v36 = (unsigned int)a4 & ((_DWORD)v36 + v35);
          v10 = (__int64 *)(a5 + 16 * v36);
          v37 = *v10;
          if ( v15 == *v10 )
            goto LABEL_21;
          ++v35;
        }
        goto LABEL_37;
      }
LABEL_19:
      sub_29E9530(v88, 2 * v27);
      v28 = *(_DWORD *)(a1 + 72);
      if ( !v28 )
        goto LABEL_165;
      a4 = (unsigned int)(v28 - 1);
      a5 = *(_QWORD *)(a1 + 56);
      LODWORD(v29) = a4 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v30 = *(_DWORD *)(a1 + 64) + 1;
      v10 = (__int64 *)(a5 + 16LL * (unsigned int)v29);
      v31 = *v10;
      if ( v15 == *v10 )
        goto LABEL_21;
      v86 = 1;
      v8 = 0;
      while ( v31 != -4096 )
      {
        if ( !v8 && v31 == -8192 )
          v8 = (__int64)v10;
        v29 = (unsigned int)a4 & ((_DWORD)v29 + v86);
        v10 = (__int64 *)(a5 + 16 * v29);
        v31 = *v10;
        if ( v15 == *v10 )
          goto LABEL_21;
        ++v86;
      }
LABEL_37:
      if ( v8 )
        v10 = (__int64 *)v8;
LABEL_21:
      *(_DWORD *)(a1 + 64) = v30;
      if ( *v10 != -4096 )
        --*(_DWORD *)(a1 + 68);
      *v10 = v15;
      v32 = v98;
      v13 = v10 + 1;
      v10[1] = 0;
      v14 = (__int64)v32[(unsigned int)v99 - 1];
      goto LABEL_6;
    }
LABEL_4:
    v12 = v11[1];
    v13 = v11 + 1;
    v14 = (__int64)v98[(unsigned int)v99 - 1];
    if ( v12 )
      sub_B91220((__int64)v13, v12);
LABEL_6:
    *v13 = v14;
    if ( v14 )
      sub_B96E90((__int64)v13, v14, 1);
    ++v5;
  }
  while ( v6 != v5 );
  v50 = *(__int64 **)(a1 + 32);
  a3 = *(unsigned int *)(a1 + 40);
  v95 = (__int64 *)v97;
  a2 = (__int64)&v50[a3];
  v89 = (__int64 *)a2;
  v96 = 0x400000000LL;
  if ( v50 == (__int64 *)a2 )
    goto LABEL_49;
  v94 = v50;
  while ( 2 )
  {
    v38 = *v94;
    v51 = *(_BYTE *)(*v94 - 16);
    if ( (v51 & 2) != 0 )
    {
      v52 = *(_QWORD **)(v38 - 32);
      v53 = *(unsigned int *)(v38 - 24);
    }
    else
    {
      v53 = (*(_WORD *)(v38 - 16) >> 6) & 0xF;
      v52 = (_QWORD *)(v38 + -16 - 8LL * ((v51 >> 2) & 0xF));
    }
    v54 = &v52[v53];
    if ( v54 != v52 )
    {
      v92 = *v94;
      while ( 1 )
      {
        v60 = (_BYTE *)*v52;
        if ( (unsigned __int8)(*(_BYTE *)*v52 - 5) <= 0x1Fu )
        {
          v63 = *(_DWORD *)(a1 + 72);
          if ( !v63 )
          {
            ++*(_QWORD *)(a1 + 48);
            goto LABEL_71;
          }
          v8 = v63 - 1;
          a5 = *(_QWORD *)(a1 + 56);
          v55 = 1;
          v56 = 0;
          v57 = v8 & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
          v58 = (__int64 *)(a5 + 16LL * v57);
          v59 = *v58;
          if ( v60 != (_BYTE *)*v58 )
          {
            while ( v59 != -4096 )
            {
              if ( !v56 && v59 == -8192 )
                v56 = v58;
              v57 = v8 & (v55 + v57);
              v58 = (__int64 *)(a5 + 16LL * v57);
              v59 = *v58;
              if ( v60 == (_BYTE *)*v58 )
                goto LABEL_64;
              ++v55;
            }
            if ( !v56 )
              v56 = v58;
            v69 = *(_DWORD *)(a1 + 64);
            ++*(_QWORD *)(a1 + 48);
            v68 = v69 + 1;
            if ( 4 * (v69 + 1) >= 3 * v63 )
            {
LABEL_71:
              sub_29E9530(v88, 2 * v63);
              v64 = *(_DWORD *)(a1 + 72);
              if ( !v64 )
                goto LABEL_163;
              v65 = v64 - 1;
              v66 = *(_QWORD *)(a1 + 56);
              v67 = (v64 - 1) & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
              v68 = *(_DWORD *)(a1 + 64) + 1;
              v56 = (__int64 *)(v66 + 16LL * v67);
              a5 = *v56;
              if ( v60 != (_BYTE *)*v56 )
              {
                v84 = 1;
                v85 = 0;
                while ( a5 != -4096 )
                {
                  if ( a5 != -8192 || v85 )
                    v56 = v85;
                  v67 = v65 & (v84 + v67);
                  v8 = v66 + 16LL * v67;
                  a5 = *(_QWORD *)v8;
                  if ( v60 == *(_BYTE **)v8 )
                  {
                    v56 = (__int64 *)(v66 + 16LL * v67);
                    goto LABEL_73;
                  }
                  ++v84;
                  v85 = v56;
                  v56 = (__int64 *)(v66 + 16LL * v67);
                }
                if ( v85 )
                  v56 = v85;
              }
            }
            else if ( v63 - *(_DWORD *)(a1 + 68) - v68 <= v63 >> 3 )
            {
              sub_29E9530(v88, v63);
              v70 = *(_DWORD *)(a1 + 72);
              if ( !v70 )
              {
LABEL_163:
                ++*(_DWORD *)(a1 + 64);
                BUG();
              }
              v71 = v70 - 1;
              v72 = *(_QWORD *)(a1 + 56);
              a5 = 0;
              v73 = v71 & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
              v8 = 1;
              v68 = *(_DWORD *)(a1 + 64) + 1;
              v56 = (__int64 *)(v72 + 16LL * v73);
              v74 = *v56;
              if ( v60 != (_BYTE *)*v56 )
              {
                while ( v74 != -4096 )
                {
                  if ( v74 == -8192 && !a5 )
                    a5 = (__int64)v56;
                  v87 = v8 + 1;
                  v8 = v71 & (v73 + (unsigned int)v8);
                  v73 = v8;
                  v56 = (__int64 *)(v72 + 16LL * (unsigned int)v8);
                  v74 = *v56;
                  if ( v60 == (_BYTE *)*v56 )
                    goto LABEL_73;
                  v8 = v87;
                }
                if ( a5 )
                  v56 = (__int64 *)a5;
              }
            }
LABEL_73:
            *(_DWORD *)(a1 + 64) = v68;
            if ( *v56 != -4096 )
              --*(_DWORD *)(a1 + 68);
            *v56 = (__int64)v60;
            v60 = 0;
            v56[1] = 0;
            goto LABEL_65;
          }
LABEL_64:
          v60 = (_BYTE *)v58[1];
        }
LABEL_65:
        v61 = (unsigned int)v96;
        v62 = (unsigned int)v96 + 1LL;
        if ( v62 > HIDWORD(v96) )
        {
          sub_C8D5F0((__int64)&v95, v97, v62, 8u, a5, v8);
          v61 = (unsigned int)v96;
        }
        ++v52;
        v95[v61] = (__int64)v60;
        LODWORD(v96) = v96 + 1;
        if ( v54 == v52 )
        {
          v38 = v92;
          break;
        }
      }
    }
    v39 = (__int64 *)(*(_QWORD *)(v38 + 8) & 0xFFFFFFFFFFFFFFF8LL);
    if ( (*(_QWORD *)(v38 + 8) & 4) != 0 )
      v39 = (__int64 *)*v39;
    v91 = v38;
    v40 = sub_B9C770(v39, v95, (__int64 *)(unsigned int)v96, 0, 1);
    a2 = *(unsigned int *)(a1 + 72);
    v41 = v91;
    v42 = (__int64 *)v40;
    if ( !(_DWORD)a2 )
    {
      ++*(_QWORD *)(a1 + 48);
      goto LABEL_112;
    }
    v8 = (unsigned int)(a2 - 1);
    a5 = *(_QWORD *)(a1 + 56);
    v43 = 1;
    a3 = 0;
    v44 = v8 & (((unsigned int)v91 >> 9) ^ ((unsigned int)v91 >> 4));
    v45 = (__int64 *)(a5 + 16LL * v44);
    a4 = *v45;
    if ( v91 != *v45 )
    {
      while ( a4 != -4096 )
      {
        if ( !a3 && a4 == -8192 )
          a3 = (__int64)v45;
        v44 = v8 & (v43 + v44);
        v45 = (__int64 *)(a5 + 16LL * v44);
        a4 = *v45;
        if ( v91 == *v45 )
          goto LABEL_45;
        ++v43;
      }
      if ( !a3 )
        a3 = (__int64)v45;
      v75 = *(_DWORD *)(a1 + 64);
      ++*(_QWORD *)(a1 + 48);
      v76 = v75 + 1;
      if ( 4 * v76 < (unsigned int)(3 * a2) )
      {
        a4 = (unsigned int)(a2 - *(_DWORD *)(a1 + 68) - v76);
        if ( (unsigned int)a4 <= (unsigned int)a2 >> 3 )
        {
          sub_29E9530(v88, a2);
          v81 = *(_DWORD *)(a1 + 72);
          if ( !v81 )
          {
LABEL_164:
            ++*(_DWORD *)(a1 + 64);
            BUG();
          }
          a5 = (unsigned int)(v81 - 1);
          a2 = *(_QWORD *)(a1 + 56);
          v8 = 0;
          v82 = a5 & (((unsigned int)v91 >> 9) ^ ((unsigned int)v91 >> 4));
          v41 = v91;
          v83 = 1;
          v76 = *(_DWORD *)(a1 + 64) + 1;
          a3 = a2 + 16LL * v82;
          a4 = *(_QWORD *)a3;
          if ( v91 != *(_QWORD *)a3 )
          {
            while ( a4 != -4096 )
            {
              if ( a4 == -8192 && !v8 )
                v8 = a3;
              v82 = a5 & (v83 + v82);
              a3 = a2 + 16LL * v82;
              a4 = *(_QWORD *)a3;
              if ( v91 == *(_QWORD *)a3 )
                goto LABEL_102;
              ++v83;
            }
            if ( v8 )
              a3 = v8;
          }
        }
LABEL_102:
        *(_DWORD *)(a1 + 64) = v76;
        if ( *(_QWORD *)a3 != -4096 )
          --*(_DWORD *)(a1 + 68);
        *(_QWORD *)a3 = v41;
        *(_QWORD *)(a3 + 8) = 0;
        v46 = MEMORY[8];
        if ( (MEMORY[8] & 4) != 0 )
          goto LABEL_105;
        goto LABEL_46;
      }
LABEL_112:
      sub_29E9530(v88, 2 * a2);
      v77 = *(_DWORD *)(a1 + 72);
      if ( !v77 )
        goto LABEL_164;
      v41 = v91;
      v8 = (unsigned int)(v77 - 1);
      v78 = *(_QWORD *)(a1 + 56);
      a4 = (unsigned int)v8 & (((unsigned int)v91 >> 9) ^ ((unsigned int)v91 >> 4));
      v76 = *(_DWORD *)(a1 + 64) + 1;
      a3 = v78 + 16 * a4;
      a2 = *(_QWORD *)a3;
      if ( v91 != *(_QWORD *)a3 )
      {
        v79 = 1;
        v80 = 0;
        while ( a2 != -4096 )
        {
          if ( a2 == -8192 && !v80 )
            v80 = (__int64 *)a3;
          a5 = (unsigned int)(v79 + 1);
          a4 = (unsigned int)v8 & (v79 + (_DWORD)a4);
          a3 = v78 + 16LL * (unsigned int)a4;
          a2 = *(_QWORD *)a3;
          if ( v91 == *(_QWORD *)a3 )
            goto LABEL_102;
          ++v79;
        }
        if ( v80 )
          a3 = (__int64)v80;
      }
      goto LABEL_102;
    }
LABEL_45:
    v46 = *(_QWORD *)(v45[1] + 8);
    if ( (v46 & 4) == 0 )
      goto LABEL_46;
LABEL_105:
    a2 = (__int64)v42;
    sub_BA6110((const __m128i *)(v46 & 0xFFFFFFFFFFFFFFF8LL), v42);
LABEL_46:
    ++v94;
    LODWORD(v96) = 0;
    if ( v89 != v94 )
      continue;
    break;
  }
  if ( v95 != (__int64 *)v97 )
    _libc_free((unsigned __int64)v95);
LABEL_49:
  v47 = v98;
  v48 = &v98[(unsigned int)v99];
  if ( v98 != v48 )
  {
    do
    {
      v49 = (__int64)*--v48;
      if ( v49 )
        sub_BA65D0(v49, a2, a3, a4, a5);
    }
    while ( v47 != v48 );
    v48 = v98;
  }
  if ( v48 != (__int64 **)v100 )
    _libc_free((unsigned __int64)v48);
}
