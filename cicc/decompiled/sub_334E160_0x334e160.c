// Function: sub_334E160
// Address: 0x334e160
//
void __fastcall sub_334E160(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // r15
  unsigned int v10; // r14d
  __int64 v11; // rax
  __int64 v12; // rdx
  int v13; // esi
  __int64 v14; // rsi
  __int64 v15; // r12
  __int64 v16; // r13
  unsigned __int64 v17; // rax
  __int64 *v18; // r13
  unsigned int v19; // r11d
  __int64 *v20; // r15
  unsigned int v21; // esi
  __int64 v22; // rbx
  int v23; // r10d
  __int64 v24; // r8
  _QWORD *v25; // rdx
  unsigned int v26; // edi
  _QWORD *v27; // rax
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // rax
  int v31; // edx
  __int64 v32; // rcx
  bool v33; // zf
  const void *v34; // r13
  __int64 v35; // r15
  __int64 v36; // rbx
  __int64 v37; // rax
  char *v38; // r14
  signed __int64 v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // rax
  unsigned __int64 v42; // rdx
  unsigned int v43; // esi
  __int64 v44; // rdi
  __int64 v45; // r10
  unsigned int v46; // edx
  __int64 *v47; // rax
  __int64 v48; // rcx
  int v49; // edx
  int v50; // eax
  __int64 v51; // r10
  __int64 *v52; // rsi
  unsigned int v53; // ecx
  int v54; // edi
  int v55; // eax
  int v56; // eax
  int v57; // esi
  int v58; // esi
  __int64 v59; // r8
  unsigned int v60; // ecx
  __int64 v61; // rdi
  int v62; // r10d
  _QWORD *v63; // r9
  int v64; // ecx
  int v65; // ecx
  __int64 v66; // rdi
  _QWORD *v67; // r8
  unsigned int v68; // r14d
  int v69; // r9d
  __int64 v70; // rsi
  int v71; // eax
  __int64 v72; // r10
  unsigned int v73; // ecx
  int v74; // edi
  unsigned int v75; // [rsp+Ch] [rbp-94h]
  unsigned int v76; // [rsp+Ch] [rbp-94h]
  unsigned int v77; // [rsp+Ch] [rbp-94h]
  __int64 v79; // [rsp+18h] [rbp-88h]
  __int64 *v80; // [rsp+20h] [rbp-80h] BYREF
  __int64 v81; // [rsp+28h] [rbp-78h]
  _BYTE v82[112]; // [rsp+30h] [rbp-70h] BYREF

  v6 = a1;
  v80 = (__int64 *)v82;
  v81 = 0x800000000LL;
  v7 = *(_QWORD *)(a1 + 592);
  v8 = *(_QWORD *)(v7 + 408);
  v9 = v7 + 400;
  if ( v8 == v7 + 400 )
    goto LABEL_33;
  v10 = 0;
  do
  {
    if ( !v8 )
      BUG();
    v11 = *(_QWORD *)(v8 + 48);
    if ( v11 )
    {
      v12 = 0;
      do
      {
        v11 = *(_QWORD *)(v11 + 32);
        ++v12;
      }
      while ( v11 );
    }
    else
    {
      v12 = 0;
    }
    v13 = *(_DWORD *)(v8 + 60);
    *(_DWORD *)(v8 + 28) = v12;
    if ( v13 )
    {
      v14 = (unsigned int)(v13 - 1);
      if ( *(_WORD *)(*(_QWORD *)(v8 + 40) + 16 * v14) == 262 )
      {
        v15 = v8 - 8;
        v16 = v8 - 8;
        if ( (unsigned __int8)sub_33CF8A0(v8 - 8, v14, v12, a4, a5, a6) )
        {
          while ( 1 )
          {
            v40 = *(_QWORD *)(v16 + 56);
            if ( !v40 )
              break;
            while ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v40 + 48LL) + 16LL * *(unsigned int *)(v40 + 8)) != 262 )
            {
              v40 = *(_QWORD *)(v40 + 32);
              if ( !v40 )
                goto LABEL_41;
            }
            if ( !*(_QWORD *)(v40 + 16) )
              break;
            v16 = *(_QWORD *)(v40 + 16);
          }
LABEL_41:
          v41 = (unsigned int)v81;
          v42 = (unsigned int)v81 + 1LL;
          if ( v42 > HIDWORD(v81) )
          {
            sub_C8D5F0((__int64)&v80, v82, v42, 8u, a5, a6);
            v41 = (unsigned int)v81;
          }
          v80[v41] = v15;
          LODWORD(v81) = v81 + 1;
          v43 = *(_DWORD *)(a1 + 680);
          v44 = a1 + 656;
          if ( v43 )
          {
            v45 = *(_QWORD *)(a1 + 664);
            v75 = ((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4);
            v46 = (v43 - 1) & v75;
            v47 = (__int64 *)(v45 + 16LL * v46);
            v48 = *v47;
            if ( v15 == *v47 )
              goto LABEL_10;
            a5 = 1;
            a6 = 0;
            while ( v48 != -4096 )
            {
              if ( !a6 && v48 == -8192 )
                a6 = (__int64)v47;
              v46 = (v43 - 1) & (a5 + v46);
              v47 = (__int64 *)(v45 + 16LL * v46);
              v48 = *v47;
              if ( v15 == *v47 )
                goto LABEL_10;
              a5 = (unsigned int)(a5 + 1);
            }
            if ( a6 )
              v47 = (__int64 *)a6;
            ++*(_QWORD *)(a1 + 656);
            v49 = *(_DWORD *)(a1 + 672) + 1;
            if ( 4 * v49 < 3 * v43 )
            {
              a6 = v43 >> 3;
              if ( v43 - *(_DWORD *)(a1 + 676) - v49 > (unsigned int)a6 )
                goto LABEL_94;
              sub_334DF80(v44, v43);
              v50 = *(_DWORD *)(a1 + 680);
              if ( !v50 )
                goto LABEL_125;
              a5 = (unsigned int)(v50 - 1);
              v51 = *(_QWORD *)(a1 + 664);
              v52 = 0;
              v53 = a5 & v75;
              v49 = *(_DWORD *)(a1 + 672) + 1;
              v54 = 1;
              v47 = (__int64 *)(v51 + 16LL * ((unsigned int)a5 & v75));
              a6 = *v47;
              if ( v15 == *v47 )
                goto LABEL_94;
              while ( a6 != -4096 )
              {
                if ( !v52 && a6 == -8192 )
                  v52 = v47;
                v53 = a5 & (v54 + v53);
                v47 = (__int64 *)(v51 + 16LL * v53);
                a6 = *v47;
                if ( v15 == *v47 )
                  goto LABEL_94;
                ++v54;
              }
LABEL_54:
              if ( v52 )
                v47 = v52;
LABEL_94:
              *(_DWORD *)(a1 + 672) = v49;
              if ( *v47 != -4096 )
                --*(_DWORD *)(a1 + 676);
              *v47 = v15;
              v47[1] = v16;
              goto LABEL_10;
            }
          }
          else
          {
            ++*(_QWORD *)(a1 + 656);
          }
          sub_334DF80(v44, 2 * v43);
          v71 = *(_DWORD *)(a1 + 680);
          if ( !v71 )
          {
LABEL_125:
            ++*(_DWORD *)(a1 + 672);
            BUG();
          }
          a5 = (unsigned int)(v71 - 1);
          v72 = *(_QWORD *)(a1 + 664);
          v73 = a5 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
          v49 = *(_DWORD *)(a1 + 672) + 1;
          v47 = (__int64 *)(v72 + 16LL * v73);
          a6 = *v47;
          if ( v15 == *v47 )
            goto LABEL_94;
          v74 = 1;
          v52 = 0;
          while ( a6 != -4096 )
          {
            if ( a6 == -8192 && !v52 )
              v52 = v47;
            v73 = a5 & (v74 + v73);
            v47 = (__int64 *)(v72 + 16LL * v73);
            a6 = *v47;
            if ( v15 == *v47 )
              goto LABEL_94;
            ++v74;
          }
          goto LABEL_54;
        }
      }
    }
LABEL_10:
    a4 = *(unsigned int *)(v8 + 16);
    if ( (int)a4 < 0 )
      goto LABEL_71;
    v17 = (0x3FF8000FFE42uLL >> a4) & 1;
    if ( (int)a4 >= 46 )
      LOBYTE(v17) = 0;
    if ( (_DWORD)a4 != 324 && !(_BYTE)v17 )
LABEL_71:
      ++v10;
    v8 = *(_QWORD *)(v8 + 8);
  }
  while ( v9 != v8 );
  v18 = v80;
  v19 = v10;
  v6 = a1;
  v20 = &v80[(unsigned int)v81];
  if ( v20 != v80 )
  {
    v79 = a1 + 656;
    while ( 1 )
    {
      v21 = *(_DWORD *)(v6 + 680);
      v22 = *v18;
      if ( !v21 )
        break;
      v23 = 1;
      v24 = *(_QWORD *)(v6 + 664);
      v25 = 0;
      v26 = (v21 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v27 = (_QWORD *)(v24 + 16LL * v26);
      v28 = *v27;
      if ( v22 != *v27 )
      {
        while ( v28 != -4096 )
        {
          if ( !v25 && v28 == -8192 )
            v25 = v27;
          v26 = (v21 - 1) & (v23 + v26);
          v27 = (_QWORD *)(v24 + 16LL * v26);
          v28 = *v27;
          if ( v22 == *v27 )
            goto LABEL_20;
          ++v23;
        }
        if ( !v25 )
          v25 = v27;
        v55 = *(_DWORD *)(v6 + 672);
        ++*(_QWORD *)(v6 + 656);
        v56 = v55 + 1;
        if ( 4 * v56 < 3 * v21 )
        {
          if ( v21 - *(_DWORD *)(v6 + 676) - v56 <= v21 >> 3 )
          {
            v77 = v19;
            sub_334DF80(v79, v21);
            v64 = *(_DWORD *)(v6 + 680);
            if ( !v64 )
            {
LABEL_126:
              ++*(_DWORD *)(v6 + 672);
              BUG();
            }
            v65 = v64 - 1;
            v66 = *(_QWORD *)(v6 + 664);
            v67 = 0;
            v68 = v65 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
            v19 = v77;
            v69 = 1;
            v56 = *(_DWORD *)(v6 + 672) + 1;
            v25 = (_QWORD *)(v66 + 16LL * v68);
            v70 = *v25;
            if ( v22 != *v25 )
            {
              while ( v70 != -4096 )
              {
                if ( v70 == -8192 && !v67 )
                  v67 = v25;
                v68 = v65 & (v69 + v68);
                v25 = (_QWORD *)(v66 + 16LL * v68);
                v70 = *v25;
                if ( v22 == *v25 )
                  goto LABEL_67;
                ++v69;
              }
              if ( v67 )
                v25 = v67;
            }
          }
          goto LABEL_67;
        }
LABEL_77:
        v76 = v19;
        sub_334DF80(v79, 2 * v21);
        v57 = *(_DWORD *)(v6 + 680);
        if ( !v57 )
          goto LABEL_126;
        v58 = v57 - 1;
        v59 = *(_QWORD *)(v6 + 664);
        v19 = v76;
        v60 = v58 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
        v56 = *(_DWORD *)(v6 + 672) + 1;
        v25 = (_QWORD *)(v59 + 16LL * v60);
        v61 = *v25;
        if ( v22 != *v25 )
        {
          v62 = 1;
          v63 = 0;
          while ( v61 != -4096 )
          {
            if ( v61 == -8192 && !v63 )
              v63 = v25;
            v60 = v58 & (v62 + v60);
            v25 = (_QWORD *)(v59 + 16LL * v60);
            v61 = *v25;
            if ( v22 == *v25 )
              goto LABEL_67;
            ++v62;
          }
          if ( v63 )
            v25 = v63;
        }
LABEL_67:
        *(_DWORD *)(v6 + 672) = v56;
        if ( *v25 != -4096 )
          --*(_DWORD *)(v6 + 676);
        *v25 = v22;
        v29 = 0;
        v25[1] = 0;
        goto LABEL_21;
      }
LABEL_20:
      v29 = v27[1];
LABEL_21:
      v30 = *(_QWORD *)(v22 + 56);
      v31 = *(_DWORD *)(v22 + 36);
      if ( v30 )
      {
        v32 = *(_QWORD *)(v22 + 56);
        while ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v32 + 48LL) + 16LL * *(unsigned int *)(v32 + 8)) != 262 )
        {
          v32 = *(_QWORD *)(v32 + 32);
          if ( !v32 )
            goto LABEL_25;
        }
        v32 = *(_QWORD *)(v32 + 16);
        do
        {
LABEL_25:
          v33 = v32 == *(_QWORD *)(v30 + 16);
          v30 = *(_QWORD *)(v30 + 32);
          v31 -= v33;
        }
        while ( v30 );
      }
      ++v18;
      *(_DWORD *)(v29 + 36) += v31;
      *(_DWORD *)(v22 + 36) = 1;
      if ( v20 == v18 )
        goto LABEL_27;
    }
    ++*(_QWORD *)(v6 + 656);
    goto LABEL_77;
  }
LABEL_27:
  v34 = *(const void **)(v6 + 632);
  if ( (__int64)(*(_QWORD *)(v6 + 648) - (_QWORD)v34) >> 3 < (unsigned __int64)v19 )
  {
    v35 = 8LL * v19;
    v36 = *(_QWORD *)(v6 + 640) - (_QWORD)v34;
    if ( v19 )
    {
      v37 = sub_22077B0(8LL * v19);
      v34 = *(const void **)(v6 + 632);
      v38 = (char *)v37;
      v39 = *(_QWORD *)(v6 + 640) - (_QWORD)v34;
      if ( v39 <= 0 )
        goto LABEL_30;
LABEL_74:
      memmove(v38, v34, v39);
      goto LABEL_75;
    }
    v39 = *(_QWORD *)(v6 + 640) - (_QWORD)v34;
    v38 = 0;
    if ( v36 > 0 )
      goto LABEL_74;
LABEL_30:
    if ( v34 )
LABEL_75:
      j_j___libc_free_0((unsigned __int64)v34);
    *(_QWORD *)(v6 + 632) = v38;
    *(_QWORD *)(v6 + 640) = &v38[v36];
    *(_QWORD *)(v6 + 648) = &v38[v35];
  }
  v7 = *(_QWORD *)(v6 + 592);
LABEL_33:
  sub_334DB50(v6, *(_QWORD *)(v7 + 384));
  if ( v80 != (__int64 *)v82 )
    _libc_free((unsigned __int64)v80);
}
