// Function: sub_37C1010
// Address: 0x37c1010
//
void __fastcall sub_37C1010(char *a1, char *a2, __int64 a3)
{
  __int64 v3; // r12
  char *v4; // rbx
  __int64 v5; // rbx
  char *v6; // r12
  __int64 v7; // r11
  unsigned int v8; // edi
  __int64 v9; // rcx
  int v10; // r10d
  __int64 *v11; // rdx
  unsigned int v12; // r9d
  __int64 *v13; // rax
  __int64 v14; // r8
  unsigned int v15; // r8d
  __int64 v16; // r13
  int v17; // r15d
  __int64 *v18; // rdx
  unsigned int v19; // r10d
  __int64 *v20; // rax
  __int64 v21; // r9
  unsigned int v22; // esi
  __int64 v23; // r15
  __int64 v24; // r13
  int v25; // esi
  int v26; // esi
  __int64 v27; // r8
  unsigned int v28; // ecx
  int v29; // eax
  __int64 v30; // rdi
  int v31; // esi
  int v32; // esi
  __int64 v33; // r8
  unsigned int v34; // ecx
  int v35; // eax
  __int64 v36; // rdi
  int v37; // eax
  int v38; // ecx
  int v39; // ecx
  __int64 *v40; // r9
  unsigned int v41; // r14d
  __int64 v42; // rdi
  int v43; // r10d
  __int64 v44; // rsi
  int v45; // eax
  int v46; // ecx
  int v47; // ecx
  __int64 *v48; // r8
  unsigned int v49; // r14d
  __int64 v50; // rdi
  int v51; // r10d
  __int64 v52; // rsi
  int v53; // r12d
  __int64 *v54; // r10
  int v55; // r14d
  __int64 *v56; // r9
  __int64 v57; // [rsp+0h] [rbp-70h]
  __int64 v58; // [rsp+0h] [rbp-70h]
  __int64 v59; // [rsp+0h] [rbp-70h]
  __int64 v60; // [rsp+0h] [rbp-70h]
  __int64 v62; // [rsp+18h] [rbp-58h]
  char *v64; // [rsp+28h] [rbp-48h]
  char *v65; // [rsp+30h] [rbp-40h]
  __int64 v66[7]; // [rsp+38h] [rbp-38h] BYREF

  v66[0] = a3;
  if ( a1 == a2 || a2 == a1 + 8 )
    return;
  v64 = a1 + 8;
  do
  {
    while ( sub_37C0D30(v66, *(__int64 **)(*(_QWORD *)v64 + 80LL), *(_QWORD *)a1) )
    {
      v3 = *(_QWORD *)v64;
      v4 = v64 + 8;
      if ( a1 != v64 )
        memmove(a1 + 8, a1, v64 - a1);
      v64 += 8;
      *(_QWORD *)a1 = v3;
      if ( a2 == v4 )
        return;
    }
    v5 = v66[0];
    v6 = v64;
    v7 = *(_QWORD *)v64;
    v62 = v66[0] + 664;
    while ( 1 )
    {
      v22 = *(_DWORD *)(v5 + 688);
      v65 = v6;
      v23 = *((_QWORD *)v6 - 1);
      v24 = **(_QWORD **)(v7 + 80);
      if ( v22 )
      {
        v8 = v22 - 1;
        v9 = *(_QWORD *)(v5 + 672);
        v10 = 1;
        v11 = 0;
        v12 = (v22 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
        v13 = (__int64 *)(v9 + 16LL * v12);
        v14 = *v13;
        if ( v24 == *v13 )
        {
LABEL_10:
          v15 = *((_DWORD *)v13 + 2);
          v16 = **(_QWORD **)(v23 + 80);
          goto LABEL_11;
        }
        while ( v14 != -4096 )
        {
          if ( !v11 && v14 == -8192 )
            v11 = v13;
          v12 = v8 & (v10 + v12);
          v13 = (__int64 *)(v9 + 16LL * v12);
          v14 = *v13;
          if ( v24 == *v13 )
            goto LABEL_10;
          ++v10;
        }
        if ( !v11 )
          v11 = v13;
        v45 = *(_DWORD *)(v5 + 680);
        ++*(_QWORD *)(v5 + 664);
        v29 = v45 + 1;
        if ( 4 * v29 < 3 * v22 )
        {
          if ( v22 - *(_DWORD *)(v5 + 684) - v29 <= v22 >> 3 )
          {
            v60 = v7;
            sub_2E515B0(v62, v22);
            v46 = *(_DWORD *)(v5 + 688);
            if ( !v46 )
            {
LABEL_93:
              ++*(_DWORD *)(v5 + 680);
              BUG();
            }
            v47 = v46 - 1;
            v48 = 0;
            v7 = v60;
            v49 = v47 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
            v50 = *(_QWORD *)(v5 + 672);
            v51 = 1;
            v29 = *(_DWORD *)(v5 + 680) + 1;
            v11 = (__int64 *)(v50 + 16LL * v49);
            v52 = *v11;
            if ( v24 != *v11 )
            {
              while ( v52 != -4096 )
              {
                if ( v52 == -8192 && !v48 )
                  v48 = v11;
                v49 = v47 & (v51 + v49);
                v11 = (__int64 *)(v50 + 16LL * v49);
                v52 = *v11;
                if ( v24 == *v11 )
                  goto LABEL_18;
                ++v51;
              }
              if ( v48 )
                v11 = v48;
            }
          }
          goto LABEL_18;
        }
      }
      else
      {
        ++*(_QWORD *)(v5 + 664);
      }
      v57 = v7;
      sub_2E515B0(v62, 2 * v22);
      v25 = *(_DWORD *)(v5 + 688);
      if ( !v25 )
        goto LABEL_93;
      v26 = v25 - 1;
      v27 = *(_QWORD *)(v5 + 672);
      v7 = v57;
      v28 = v26 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v29 = *(_DWORD *)(v5 + 680) + 1;
      v11 = (__int64 *)(v27 + 16LL * v28);
      v30 = *v11;
      if ( v24 != *v11 )
      {
        v55 = 1;
        v56 = 0;
        while ( v30 != -4096 )
        {
          if ( !v56 && v30 == -8192 )
            v56 = v11;
          v28 = v26 & (v55 + v28);
          v11 = (__int64 *)(v27 + 16LL * v28);
          v30 = *v11;
          if ( v24 == *v11 )
            goto LABEL_18;
          ++v55;
        }
        if ( v56 )
          v11 = v56;
      }
LABEL_18:
      *(_DWORD *)(v5 + 680) = v29;
      if ( *v11 != -4096 )
        --*(_DWORD *)(v5 + 684);
      *v11 = v24;
      *((_DWORD *)v11 + 2) = 0;
      v22 = *(_DWORD *)(v5 + 688);
      v16 = **(_QWORD **)(v23 + 80);
      if ( !v22 )
      {
        ++*(_QWORD *)(v5 + 664);
        goto LABEL_22;
      }
      v9 = *(_QWORD *)(v5 + 672);
      v8 = v22 - 1;
      v15 = 0;
LABEL_11:
      v17 = 1;
      v18 = 0;
      v19 = v8 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v20 = (__int64 *)(v9 + 16LL * v19);
      v21 = *v20;
      if ( *v20 != v16 )
        break;
LABEL_12:
      v6 -= 8;
      if ( v15 >= *((_DWORD *)v20 + 2) )
        goto LABEL_27;
      *((_QWORD *)v6 + 1) = *(_QWORD *)v6;
    }
    while ( v21 != -4096 )
    {
      if ( !v18 && v21 == -8192 )
        v18 = v20;
      v19 = v8 & (v17 + v19);
      v20 = (__int64 *)(v9 + 16LL * v19);
      v21 = *v20;
      if ( *v20 == v16 )
        goto LABEL_12;
      ++v17;
    }
    if ( !v18 )
      v18 = v20;
    v37 = *(_DWORD *)(v5 + 680);
    ++*(_QWORD *)(v5 + 664);
    v35 = v37 + 1;
    if ( 4 * v35 >= 3 * v22 )
    {
LABEL_22:
      v58 = v7;
      sub_2E515B0(v62, 2 * v22);
      v31 = *(_DWORD *)(v5 + 688);
      if ( !v31 )
        goto LABEL_92;
      v32 = v31 - 1;
      v33 = *(_QWORD *)(v5 + 672);
      v7 = v58;
      v34 = v32 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v35 = *(_DWORD *)(v5 + 680) + 1;
      v18 = (__int64 *)(v33 + 16LL * v34);
      v36 = *v18;
      if ( *v18 != v16 )
      {
        v53 = 1;
        v54 = 0;
        while ( v36 != -4096 )
        {
          if ( !v54 && v36 == -8192 )
            v54 = v18;
          v34 = v32 & (v53 + v34);
          v18 = (__int64 *)(v33 + 16LL * v34);
          v36 = *v18;
          if ( *v18 == v16 )
            goto LABEL_24;
          ++v53;
        }
        if ( v54 )
          v18 = v54;
      }
    }
    else if ( v22 - (v35 + *(_DWORD *)(v5 + 684)) <= v22 >> 3 )
    {
      v59 = v7;
      sub_2E515B0(v62, v22);
      v38 = *(_DWORD *)(v5 + 688);
      if ( v38 )
      {
        v39 = v38 - 1;
        v40 = 0;
        v7 = v59;
        v41 = v39 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v42 = *(_QWORD *)(v5 + 672);
        v43 = 1;
        v35 = *(_DWORD *)(v5 + 680) + 1;
        v18 = (__int64 *)(v42 + 16LL * v41);
        v44 = *v18;
        if ( *v18 != v16 )
        {
          while ( v44 != -4096 )
          {
            if ( !v40 && v44 == -8192 )
              v40 = v18;
            v41 = v39 & (v43 + v41);
            v18 = (__int64 *)(v42 + 16LL * v41);
            v44 = *v18;
            if ( v16 == *v18 )
              goto LABEL_24;
            ++v43;
          }
          if ( v40 )
            v18 = v40;
        }
        goto LABEL_24;
      }
LABEL_92:
      ++*(_DWORD *)(v5 + 680);
      BUG();
    }
LABEL_24:
    *(_DWORD *)(v5 + 680) = v35;
    if ( *v18 != -4096 )
      --*(_DWORD *)(v5 + 684);
    *v18 = v16;
    *((_DWORD *)v18 + 2) = 0;
LABEL_27:
    *(_QWORD *)v65 = v7;
    v64 += 8;
  }
  while ( a2 != v64 );
}
