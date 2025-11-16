// Function: sub_37C0740
// Address: 0x37c0740
//
void __fastcall sub_37C0740(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 *v3; // r14
  __int64 v5; // r12
  __int64 *v6; // rbx
  __int64 v7; // rax
  unsigned int v8; // ebx
  __int64 v9; // r11
  __int64 *v10; // rbx
  unsigned int v11; // edi
  __int64 v12; // rcx
  __int64 *v13; // rdx
  int v14; // r10d
  unsigned int v15; // r9d
  __int64 *v16; // rax
  __int64 v17; // r8
  unsigned int v18; // r8d
  unsigned int v19; // r10d
  __int64 *v20; // rax
  __int64 v21; // r9
  unsigned int v22; // esi
  __int64 v23; // r12
  int v24; // ecx
  int v25; // ecx
  __int64 v26; // r8
  unsigned int v27; // edi
  int v28; // eax
  __int64 v29; // rsi
  int v30; // esi
  int v31; // esi
  __int64 v32; // r8
  unsigned int v33; // ecx
  int v34; // eax
  __int64 *v35; // rdx
  __int64 v36; // rdi
  int v37; // eax
  int v38; // ecx
  int v39; // ecx
  __int64 v40; // r8
  __int64 *v41; // r10
  int v42; // r13d
  unsigned int v43; // esi
  __int64 v44; // rdi
  int v45; // eax
  int v46; // ecx
  int v47; // ecx
  __int64 *v48; // r9
  int v49; // r10d
  unsigned int v50; // r13d
  __int64 v51; // rdi
  __int64 v52; // rsi
  int v53; // ebx
  __int64 *v54; // r10
  int v55; // r13d
  __int64 v56; // [rsp+10h] [rbp-80h]
  __int64 v57; // [rsp+10h] [rbp-80h]
  int v58; // [rsp+10h] [rbp-80h]
  __int64 v61; // [rsp+38h] [rbp-58h]
  unsigned int v62; // [rsp+40h] [rbp-50h]
  __int64 v63; // [rsp+40h] [rbp-50h]
  __int64 v64; // [rsp+40h] [rbp-50h]
  __int64 *v65; // [rsp+48h] [rbp-48h]
  __int64 v66; // [rsp+50h] [rbp-40h] BYREF
  __int64 v67[7]; // [rsp+58h] [rbp-38h] BYREF

  if ( a1 == a2 || a2 == a1 + 1 )
    return;
  v3 = a1 + 1;
  v61 = a3 + 664;
  do
  {
    while ( 1 )
    {
      v7 = *a1;
      v66 = *v3;
      v67[0] = v7;
      v8 = *(_DWORD *)sub_2E51790(v61, &v66);
      if ( v8 >= *(_DWORD *)sub_2E51790(v61, v67) )
        break;
      v5 = *v3;
      v6 = v3 + 1;
      if ( a1 != v3 )
        memmove(a1 + 1, a1, (char *)v3 - (char *)a1);
      ++v3;
      *a1 = v5;
      if ( a2 == v6 )
        return;
    }
    v9 = *v3;
    v10 = v3;
    v62 = ((unsigned int)*v3 >> 9) ^ ((unsigned int)*v3 >> 4);
    while ( 1 )
    {
      v22 = *(_DWORD *)(a3 + 688);
      v23 = *(v10 - 1);
      v65 = v10;
      if ( v22 )
      {
        v11 = v22 - 1;
        v12 = *(_QWORD *)(a3 + 672);
        v13 = 0;
        v14 = 1;
        v15 = (v22 - 1) & v62;
        v16 = (__int64 *)(v12 + 16LL * v15);
        v17 = *v16;
        if ( v9 == *v16 )
        {
LABEL_10:
          v18 = *((_DWORD *)v16 + 2);
          goto LABEL_11;
        }
        while ( v17 != -4096 )
        {
          if ( v17 == -8192 && !v13 )
            v13 = v16;
          v15 = v11 & (v14 + v15);
          v16 = (__int64 *)(v12 + 16LL * v15);
          v17 = *v16;
          if ( v9 == *v16 )
            goto LABEL_10;
          ++v14;
        }
        if ( !v13 )
          v13 = v16;
        v37 = *(_DWORD *)(a3 + 680);
        ++*(_QWORD *)(a3 + 664);
        v28 = v37 + 1;
        if ( 4 * v28 < 3 * v22 )
        {
          if ( v22 - *(_DWORD *)(a3 + 684) - v28 > v22 >> 3 )
            goto LABEL_18;
          v57 = v9;
          sub_2E515B0(v61, v22);
          v38 = *(_DWORD *)(a3 + 688);
          if ( !v38 )
          {
LABEL_90:
            ++*(_DWORD *)(a3 + 680);
            BUG();
          }
          v39 = v38 - 1;
          v40 = *(_QWORD *)(a3 + 672);
          v41 = 0;
          v9 = v57;
          v42 = 1;
          v43 = v39 & v62;
          v28 = *(_DWORD *)(a3 + 680) + 1;
          v13 = (__int64 *)(v40 + 16LL * (v39 & v62));
          v44 = *v13;
          if ( v57 == *v13 )
            goto LABEL_18;
          while ( v44 != -4096 )
          {
            if ( v44 == -8192 && !v41 )
              v41 = v13;
            v43 = v39 & (v42 + v43);
            v13 = (__int64 *)(v40 + 16LL * v43);
            v44 = *v13;
            if ( v57 == *v13 )
              goto LABEL_18;
            ++v42;
          }
          goto LABEL_42;
        }
      }
      else
      {
        ++*(_QWORD *)(a3 + 664);
      }
      v56 = v9;
      sub_2E515B0(v61, 2 * v22);
      v24 = *(_DWORD *)(a3 + 688);
      if ( !v24 )
        goto LABEL_90;
      v25 = v24 - 1;
      v9 = v56;
      v26 = *(_QWORD *)(a3 + 672);
      v27 = v25 & v62;
      v28 = *(_DWORD *)(a3 + 680) + 1;
      v13 = (__int64 *)(v26 + 16LL * (v25 & v62));
      v29 = *v13;
      if ( v56 == *v13 )
        goto LABEL_18;
      v55 = 1;
      v41 = 0;
      while ( v29 != -4096 )
      {
        if ( !v41 && v29 == -8192 )
          v41 = v13;
        v27 = v25 & (v55 + v27);
        v13 = (__int64 *)(v26 + 16LL * v27);
        v29 = *v13;
        if ( v56 == *v13 )
          goto LABEL_18;
        ++v55;
      }
LABEL_42:
      if ( v41 )
        v13 = v41;
LABEL_18:
      *(_DWORD *)(a3 + 680) = v28;
      if ( *v13 != -4096 )
        --*(_DWORD *)(a3 + 684);
      *v13 = v9;
      *((_DWORD *)v13 + 2) = 0;
      v22 = *(_DWORD *)(a3 + 688);
      if ( !v22 )
      {
        ++*(_QWORD *)(a3 + 664);
        goto LABEL_22;
      }
      v12 = *(_QWORD *)(a3 + 672);
      v11 = v22 - 1;
      v18 = 0;
LABEL_11:
      v19 = v11 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
      v20 = (__int64 *)(v12 + 16LL * v19);
      v21 = *v20;
      if ( v23 != *v20 )
        break;
LABEL_12:
      --v10;
      if ( v18 >= *((_DWORD *)v20 + 2) )
        goto LABEL_27;
      v10[1] = *v10;
    }
    v58 = 1;
    v35 = 0;
    while ( v21 != -4096 )
    {
      if ( !v35 && v21 == -8192 )
        v35 = v20;
      v19 = v11 & (v58 + v19);
      v20 = (__int64 *)(v12 + 16LL * v19);
      v21 = *v20;
      if ( v23 == *v20 )
        goto LABEL_12;
      ++v58;
    }
    if ( !v35 )
      v35 = v20;
    v45 = *(_DWORD *)(a3 + 680);
    ++*(_QWORD *)(a3 + 664);
    v34 = v45 + 1;
    if ( 4 * v34 >= 3 * v22 )
    {
LABEL_22:
      v63 = v9;
      sub_2E515B0(v61, 2 * v22);
      v30 = *(_DWORD *)(a3 + 688);
      if ( !v30 )
        goto LABEL_91;
      v31 = v30 - 1;
      v32 = *(_QWORD *)(a3 + 672);
      v9 = v63;
      v33 = v31 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
      v34 = *(_DWORD *)(a3 + 680) + 1;
      v35 = (__int64 *)(v32 + 16LL * v33);
      v36 = *v35;
      if ( v23 != *v35 )
      {
        v53 = 1;
        v54 = 0;
        while ( v36 != -4096 )
        {
          if ( !v54 && v36 == -8192 )
            v54 = v35;
          v33 = v31 & (v53 + v33);
          v35 = (__int64 *)(v32 + 16LL * v33);
          v36 = *v35;
          if ( v23 == *v35 )
            goto LABEL_24;
          ++v53;
        }
        if ( v54 )
          v35 = v54;
      }
    }
    else if ( v22 - (v34 + *(_DWORD *)(a3 + 684)) <= v22 >> 3 )
    {
      v64 = v9;
      sub_2E515B0(v61, v22);
      v46 = *(_DWORD *)(a3 + 688);
      if ( v46 )
      {
        v47 = v46 - 1;
        v48 = 0;
        v9 = v64;
        v49 = 1;
        v50 = v47 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v51 = *(_QWORD *)(a3 + 672);
        v34 = *(_DWORD *)(a3 + 680) + 1;
        v35 = (__int64 *)(v51 + 16LL * v50);
        v52 = *v35;
        if ( v23 != *v35 )
        {
          while ( v52 != -4096 )
          {
            if ( v52 == -8192 && !v48 )
              v48 = v35;
            v50 = v47 & (v49 + v50);
            v35 = (__int64 *)(v51 + 16LL * v50);
            v52 = *v35;
            if ( v23 == *v35 )
              goto LABEL_24;
            ++v49;
          }
          if ( v48 )
            v35 = v48;
        }
        goto LABEL_24;
      }
LABEL_91:
      ++*(_DWORD *)(a3 + 680);
      BUG();
    }
LABEL_24:
    *(_DWORD *)(a3 + 680) = v34;
    if ( *v35 != -4096 )
      --*(_DWORD *)(a3 + 684);
    *v35 = v23;
    *((_DWORD *)v35 + 2) = 0;
LABEL_27:
    ++v3;
    *v65 = v9;
  }
  while ( a2 != v3 );
}
