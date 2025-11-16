// Function: sub_1D00290
// Address: 0x1d00290
//
void __fastcall sub_1D00290(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r15
  unsigned int v5; // r14d
  __int64 v6; // rax
  int v7; // edx
  int v8; // esi
  __int64 v9; // r12
  __int64 v10; // r13
  int v11; // r8d
  int v12; // r9d
  __int16 v13; // cx
  unsigned __int64 v14; // rax
  unsigned int v15; // r11d
  __int64 v16; // r15
  __int64 v17; // r13
  unsigned int v18; // esi
  __int64 v19; // rbx
  __int64 v20; // r8
  unsigned int v21; // edi
  _QWORD *v22; // rax
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // rax
  int v26; // edx
  __int64 v27; // rcx
  bool v28; // zf
  const void *v29; // r13
  __int64 v30; // r15
  __int64 v31; // rbx
  __int64 v32; // rax
  char *v33; // r14
  signed __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // rax
  unsigned int v37; // esi
  __int64 v38; // rdi
  __int64 v39; // r10
  unsigned int v40; // edx
  _QWORD *v41; // rax
  __int64 v42; // rcx
  int v43; // r8d
  _QWORD *v44; // r9
  int v45; // edx
  int v46; // eax
  int v47; // r8d
  __int64 v48; // r10
  _QWORD *v49; // rsi
  unsigned int v50; // ecx
  int v51; // edi
  __int64 v52; // r9
  int v53; // r10d
  _QWORD *v54; // rdx
  int v55; // eax
  int v56; // eax
  __int64 v57; // rsi
  int v58; // esi
  int v59; // esi
  __int64 v60; // r8
  unsigned int v61; // ecx
  __int64 v62; // rdi
  int v63; // r10d
  _QWORD *v64; // r9
  int v65; // ecx
  int v66; // ecx
  __int64 v67; // rdi
  _QWORD *v68; // r8
  unsigned int v69; // r14d
  int v70; // r9d
  __int64 v71; // rsi
  int v72; // edx
  __int64 *v73; // r10
  int v74; // eax
  int v75; // r8d
  __int64 v76; // r10
  unsigned int v77; // ecx
  __int64 v78; // r9
  int v79; // edi
  unsigned int v80; // [rsp+8h] [rbp-98h]
  unsigned int v81; // [rsp+8h] [rbp-98h]
  unsigned int v82; // [rsp+8h] [rbp-98h]
  __int64 v84; // [rsp+18h] [rbp-88h]
  _BYTE *v85; // [rsp+20h] [rbp-80h] BYREF
  __int64 v86; // [rsp+28h] [rbp-78h]
  _BYTE v87[112]; // [rsp+30h] [rbp-70h] BYREF

  v1 = a1;
  v85 = v87;
  v86 = 0x800000000LL;
  v2 = *(_QWORD *)(a1 + 624);
  v3 = *(_QWORD *)(v2 + 200);
  v4 = v2 + 192;
  if ( v3 == v2 + 192 )
    goto LABEL_33;
  v5 = 0;
  do
  {
    if ( !v3 )
      BUG();
    v6 = *(_QWORD *)(v3 + 40);
    if ( v6 )
    {
      v7 = 0;
      do
      {
        v6 = *(_QWORD *)(v6 + 32);
        ++v7;
      }
      while ( v6 );
    }
    else
    {
      v7 = 0;
    }
    v8 = *(_DWORD *)(v3 + 52);
    *(_DWORD *)(v3 + 20) = v7;
    if ( v8 )
    {
      if ( *(_BYTE *)(*(_QWORD *)(v3 + 32) + 16LL * (unsigned int)(v8 - 1)) == 111 )
      {
        v9 = v3 - 8;
        v10 = v3 - 8;
        if ( (unsigned __int8)sub_1D18C40(v3 - 8) )
        {
          while ( 1 )
          {
            v35 = *(_QWORD *)(v10 + 48);
            if ( !v35 )
              break;
            while ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v35 + 40LL) + 16LL * *(unsigned int *)(v35 + 8)) != 111 )
            {
              v35 = *(_QWORD *)(v35 + 32);
              if ( !v35 )
                goto LABEL_41;
            }
            if ( !*(_QWORD *)(v35 + 16) )
              break;
            v10 = *(_QWORD *)(v35 + 16);
          }
LABEL_41:
          v36 = (unsigned int)v86;
          if ( (unsigned int)v86 >= HIDWORD(v86) )
          {
            sub_16CD150((__int64)&v85, v87, 0, 8, v11, v12);
            v36 = (unsigned int)v86;
          }
          *(_QWORD *)&v85[8 * v36] = v9;
          LODWORD(v86) = v86 + 1;
          v37 = *(_DWORD *)(a1 + 712);
          v38 = a1 + 688;
          if ( v37 )
          {
            v39 = *(_QWORD *)(a1 + 696);
            v80 = ((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4);
            v40 = (v37 - 1) & v80;
            v41 = (_QWORD *)(v39 + 16LL * v40);
            v42 = *v41;
            if ( v9 == *v41 )
              goto LABEL_10;
            v43 = 1;
            v44 = 0;
            while ( v42 != -8 )
            {
              if ( v42 == -16 && !v44 )
                v44 = v41;
              v40 = (v37 - 1) & (v43 + v40);
              v41 = (_QWORD *)(v39 + 16LL * v40);
              v42 = *v41;
              if ( v9 == *v41 )
                goto LABEL_10;
              ++v43;
            }
            if ( v44 )
              v41 = v44;
            ++*(_QWORD *)(a1 + 688);
            v45 = *(_DWORD *)(a1 + 704) + 1;
            if ( 4 * v45 < 3 * v37 )
            {
              if ( v37 - *(_DWORD *)(a1 + 708) - v45 > v37 >> 3 )
                goto LABEL_94;
              sub_1D000D0(v38, v37);
              v46 = *(_DWORD *)(a1 + 712);
              if ( !v46 )
                goto LABEL_128;
              v47 = v46 - 1;
              v48 = *(_QWORD *)(a1 + 696);
              v49 = 0;
              v50 = (v46 - 1) & v80;
              v45 = *(_DWORD *)(a1 + 704) + 1;
              v51 = 1;
              v41 = (_QWORD *)(v48 + 16LL * v50);
              v52 = *v41;
              if ( v9 == *v41 )
                goto LABEL_94;
              while ( v52 != -8 )
              {
                if ( !v49 && v52 == -16 )
                  v49 = v41;
                v50 = v47 & (v51 + v50);
                v41 = (_QWORD *)(v48 + 16LL * v50);
                v52 = *v41;
                if ( v9 == *v41 )
                  goto LABEL_94;
                ++v51;
              }
LABEL_54:
              if ( v49 )
                v41 = v49;
LABEL_94:
              *(_DWORD *)(a1 + 704) = v45;
              if ( *v41 != -8 )
                --*(_DWORD *)(a1 + 708);
              *v41 = v9;
              v41[1] = v10;
              goto LABEL_10;
            }
          }
          else
          {
            ++*(_QWORD *)(a1 + 688);
          }
          sub_1D000D0(v38, 2 * v37);
          v74 = *(_DWORD *)(a1 + 712);
          if ( !v74 )
          {
LABEL_128:
            ++*(_DWORD *)(a1 + 704);
            BUG();
          }
          v75 = v74 - 1;
          v76 = *(_QWORD *)(a1 + 696);
          v77 = (v74 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
          v45 = *(_DWORD *)(a1 + 704) + 1;
          v41 = (_QWORD *)(v76 + 16LL * v77);
          v78 = *v41;
          if ( v9 == *v41 )
            goto LABEL_94;
          v79 = 1;
          v49 = 0;
          while ( v78 != -8 )
          {
            if ( v78 == -16 && !v49 )
              v49 = v41;
            v77 = v75 & (v79 + v77);
            v41 = (_QWORD *)(v76 + 16LL * v77);
            v78 = *v41;
            if ( v9 == *v41 )
              goto LABEL_94;
            ++v79;
          }
          goto LABEL_54;
        }
      }
    }
LABEL_10:
    v13 = *(_WORD *)(v3 + 16);
    if ( v13 < 0 )
      goto LABEL_58;
    v14 = (0x7FF0007FF22uLL >> v13) & 1;
    if ( v13 >= 43 )
      LOBYTE(v14) = 0;
    if ( v13 != 209 && !(_BYTE)v14 )
LABEL_58:
      ++v5;
    v3 = *(_QWORD *)(v3 + 8);
  }
  while ( v4 != v3 );
  v1 = a1;
  v15 = v5;
  if ( (_DWORD)v86 )
  {
    v16 = 8LL * (unsigned int)v86;
    v17 = 0;
    v84 = a1 + 688;
    do
    {
      v18 = *(_DWORD *)(v1 + 712);
      v19 = *(_QWORD *)&v85[v17];
      if ( v18 )
      {
        v20 = *(_QWORD *)(v1 + 696);
        v21 = (v18 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
        v22 = (_QWORD *)(v20 + 16LL * v21);
        v23 = *v22;
        if ( v19 == *v22 )
        {
          v24 = v22[1];
          goto LABEL_21;
        }
        v53 = 1;
        v54 = 0;
        while ( v23 != -8 )
        {
          if ( v54 || v23 != -16 )
            v22 = v54;
          v72 = v53 + 1;
          v21 = (v18 - 1) & (v53 + v21);
          v73 = (__int64 *)(v20 + 16LL * v21);
          v23 = *v73;
          if ( v19 == *v73 )
          {
            v24 = v73[1];
            goto LABEL_21;
          }
          v53 = v72;
          v54 = v22;
          v22 = (_QWORD *)(v20 + 16LL * v21);
        }
        if ( !v54 )
          v54 = v22;
        v55 = *(_DWORD *)(v1 + 704);
        ++*(_QWORD *)(v1 + 688);
        v56 = v55 + 1;
        if ( 4 * v56 < 3 * v18 )
        {
          if ( v18 - *(_DWORD *)(v1 + 708) - v56 <= v18 >> 3 )
          {
            v82 = v15;
            sub_1D000D0(v84, v18);
            v65 = *(_DWORD *)(v1 + 712);
            if ( !v65 )
            {
LABEL_127:
              ++*(_DWORD *)(v1 + 704);
              BUG();
            }
            v66 = v65 - 1;
            v67 = *(_QWORD *)(v1 + 696);
            v68 = 0;
            v69 = v66 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
            v15 = v82;
            v70 = 1;
            v56 = *(_DWORD *)(v1 + 704) + 1;
            v54 = (_QWORD *)(v67 + 16LL * v69);
            v71 = *v54;
            if ( v19 != *v54 )
            {
              while ( v71 != -8 )
              {
                if ( !v68 && v71 == -16 )
                  v68 = v54;
                v69 = v66 & (v70 + v69);
                v54 = (_QWORD *)(v67 + 16LL * v69);
                v71 = *v54;
                if ( v19 == *v54 )
                  goto LABEL_66;
                ++v70;
              }
              if ( v68 )
                v54 = v68;
            }
          }
          goto LABEL_66;
        }
      }
      else
      {
        ++*(_QWORD *)(v1 + 688);
      }
      v81 = v15;
      sub_1D000D0(v84, 2 * v18);
      v58 = *(_DWORD *)(v1 + 712);
      if ( !v58 )
        goto LABEL_127;
      v59 = v58 - 1;
      v60 = *(_QWORD *)(v1 + 696);
      v15 = v81;
      v61 = v59 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      v56 = *(_DWORD *)(v1 + 704) + 1;
      v54 = (_QWORD *)(v60 + 16LL * v61);
      v62 = *v54;
      if ( v19 != *v54 )
      {
        v63 = 1;
        v64 = 0;
        while ( v62 != -8 )
        {
          if ( !v64 && v62 == -16 )
            v64 = v54;
          v61 = v59 & (v63 + v61);
          v54 = (_QWORD *)(v60 + 16LL * v61);
          v62 = *v54;
          if ( v19 == *v54 )
            goto LABEL_66;
          ++v63;
        }
        if ( v64 )
          v54 = v64;
      }
LABEL_66:
      *(_DWORD *)(v1 + 704) = v56;
      if ( *v54 != -8 )
        --*(_DWORD *)(v1 + 708);
      *v54 = v19;
      v24 = 0;
      v54[1] = 0;
LABEL_21:
      v25 = *(_QWORD *)(v19 + 48);
      v26 = *(_DWORD *)(v19 + 28);
      if ( v25 )
      {
        v27 = *(_QWORD *)(v19 + 48);
        while ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v27 + 40LL) + 16LL * *(unsigned int *)(v27 + 8)) != 111 )
        {
          v27 = *(_QWORD *)(v27 + 32);
          if ( !v27 )
            goto LABEL_25;
        }
        v27 = *(_QWORD *)(v27 + 16);
        do
        {
LABEL_25:
          v28 = v27 == *(_QWORD *)(v25 + 16);
          v25 = *(_QWORD *)(v25 + 32);
          v26 -= v28;
        }
        while ( v25 );
      }
      v17 += 8;
      *(_DWORD *)(v24 + 28) += v26;
      *(_DWORD *)(v19 + 28) = 1;
    }
    while ( v16 != v17 );
  }
  v29 = *(const void **)(v1 + 664);
  if ( (__int64)(*(_QWORD *)(v1 + 680) - (_QWORD)v29) >> 3 < (unsigned __int64)v15 )
  {
    v30 = 8LL * v15;
    v31 = *(_QWORD *)(v1 + 672) - (_QWORD)v29;
    if ( v15 )
    {
      v32 = sub_22077B0(8LL * v15);
      v29 = *(const void **)(v1 + 664);
      v33 = (char *)v32;
      v34 = *(_QWORD *)(v1 + 672) - (_QWORD)v29;
      if ( v34 <= 0 )
        goto LABEL_30;
LABEL_70:
      memmove(v33, v29, v34);
      v57 = *(_QWORD *)(v1 + 680) - (_QWORD)v29;
LABEL_71:
      j_j___libc_free_0(v29, v57);
    }
    else
    {
      v34 = *(_QWORD *)(v1 + 672) - (_QWORD)v29;
      v33 = 0;
      if ( v31 > 0 )
        goto LABEL_70;
LABEL_30:
      if ( v29 )
      {
        v57 = *(_QWORD *)(v1 + 680) - (_QWORD)v29;
        goto LABEL_71;
      }
    }
    *(_QWORD *)(v1 + 664) = v33;
    *(_QWORD *)(v1 + 672) = &v33[v31];
    *(_QWORD *)(v1 + 680) = &v33[v30];
  }
  v2 = *(_QWORD *)(v1 + 624);
LABEL_33:
  sub_1CFD950(v1, *(_QWORD *)(v2 + 176));
  if ( v85 != v87 )
    _libc_free((unsigned __int64)v85);
}
