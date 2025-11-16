// Function: sub_1E222B0
// Address: 0x1e222b0
//
void __fastcall sub_1E222B0(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 *v3; // r14
  __int64 v4; // r12
  __int64 *v5; // rbx
  __int64 v6; // r11
  __int64 v7; // r15
  __int64 *v8; // rbx
  unsigned int v9; // edx
  __int64 v10; // rcx
  unsigned int v11; // r9d
  __int64 *v12; // rax
  __int64 v13; // r8
  unsigned int v14; // edi
  unsigned int v15; // r13d
  unsigned int v16; // r9d
  __int64 *v17; // rax
  __int64 v18; // r8
  unsigned int v19; // esi
  __int64 v20; // r12
  __int64 *v21; // r10
  int v22; // edx
  int v23; // edx
  __int64 v24; // r8
  unsigned int v25; // esi
  int v26; // eax
  __int64 *v27; // rdi
  __int64 v28; // rcx
  int v29; // esi
  int v30; // esi
  __int64 v31; // r8
  unsigned int v32; // ecx
  int v33; // eax
  __int64 *v34; // rdx
  __int64 v35; // rdi
  int v36; // eax
  int v37; // ecx
  int v38; // ecx
  __int64 v39; // rdi
  __int64 *v40; // r9
  unsigned int v41; // r13d
  int v42; // r10d
  __int64 v43; // rsi
  int v44; // r13d
  int v45; // eax
  int v46; // edx
  int v47; // edx
  __int64 v48; // r8
  __int64 *v49; // r9
  int v50; // r13d
  unsigned int v51; // esi
  __int64 v52; // rcx
  int v53; // ebx
  __int64 *v54; // r10
  int v55; // r13d
  int v56; // [rsp+8h] [rbp-68h]
  __int64 v57; // [rsp+10h] [rbp-60h]
  __int64 v58; // [rsp+10h] [rbp-60h]
  __int64 *v59; // [rsp+10h] [rbp-60h]
  __int64 v60; // [rsp+10h] [rbp-60h]
  __int64 v61; // [rsp+10h] [rbp-60h]
  unsigned int v64; // [rsp+30h] [rbp-40h]
  __int64 *v65; // [rsp+30h] [rbp-40h]
  __int64 v66[7]; // [rsp+38h] [rbp-38h] BYREF

  v66[0] = a3;
  if ( a1 == a2 || a1 + 1 == a2 )
    return;
  v3 = a1 + 1;
  do
  {
    while ( sub_1E20950(v66, *v3, *a1) )
    {
      v4 = *v3;
      v5 = v3 + 1;
      if ( a1 != v3 )
        memmove(a1 + 1, a1, (char *)v3 - (char *)a1);
      ++v3;
      *a1 = v4;
      if ( a2 == v5 )
        return;
    }
    v6 = *v3;
    v7 = v66[0];
    v8 = v3;
    v64 = ((unsigned int)*v3 >> 9) ^ ((unsigned int)*v3 >> 4);
    while ( 1 )
    {
      v19 = *(_DWORD *)(v7 + 24);
      v20 = *(v8 - 1);
      v21 = v8;
      if ( v19 )
      {
        v9 = v19 - 1;
        v10 = *(_QWORD *)(v7 + 8);
        v11 = (v19 - 1) & v64;
        v12 = (__int64 *)(v10 + 16LL * v11);
        v13 = *v12;
        if ( v6 == *v12 )
        {
LABEL_10:
          v14 = *((_DWORD *)v12 + 2);
          goto LABEL_11;
        }
        v44 = 1;
        v27 = 0;
        while ( v13 != -8 )
        {
          if ( v13 == -16 && !v27 )
            v27 = v12;
          v11 = v9 & (v44 + v11);
          v12 = (__int64 *)(v10 + 16LL * v11);
          v13 = *v12;
          if ( v6 == *v12 )
            goto LABEL_10;
          ++v44;
        }
        if ( !v27 )
          v27 = v12;
        v45 = *(_DWORD *)(v7 + 16);
        ++*(_QWORD *)v7;
        v26 = v45 + 1;
        if ( 4 * v26 < 3 * v19 )
        {
          if ( v19 - *(_DWORD *)(v7 + 20) - v26 > v19 >> 3 )
            goto LABEL_18;
          v61 = v6;
          sub_1E20790(v7, v19);
          v46 = *(_DWORD *)(v7 + 24);
          if ( !v46 )
          {
LABEL_94:
            ++*(_DWORD *)(v7 + 16);
            BUG();
          }
          v47 = v46 - 1;
          v48 = *(_QWORD *)(v7 + 8);
          v49 = 0;
          v6 = v61;
          v50 = 1;
          v51 = v47 & v64;
          v21 = v8;
          v26 = *(_DWORD *)(v7 + 16) + 1;
          v27 = (__int64 *)(v48 + 16LL * (v47 & v64));
          v52 = *v27;
          if ( v61 == *v27 )
            goto LABEL_18;
          while ( v52 != -8 )
          {
            if ( v52 == -16 && !v49 )
              v49 = v27;
            v51 = v47 & (v50 + v51);
            v27 = (__int64 *)(v48 + 16LL * v51);
            v52 = *v27;
            if ( v61 == *v27 )
              goto LABEL_18;
            ++v50;
          }
          goto LABEL_51;
        }
      }
      else
      {
        ++*(_QWORD *)v7;
      }
      v57 = v6;
      sub_1E20790(v7, 2 * v19);
      v22 = *(_DWORD *)(v7 + 24);
      if ( !v22 )
        goto LABEL_94;
      v23 = v22 - 1;
      v24 = *(_QWORD *)(v7 + 8);
      v6 = v57;
      v25 = v23 & v64;
      v21 = v8;
      v26 = *(_DWORD *)(v7 + 16) + 1;
      v27 = (__int64 *)(v24 + 16LL * (v23 & v64));
      v28 = *v27;
      if ( v57 == *v27 )
        goto LABEL_18;
      v55 = 1;
      v49 = 0;
      while ( v28 != -8 )
      {
        if ( !v49 && v28 == -16 )
          v49 = v27;
        v25 = v23 & (v55 + v25);
        v27 = (__int64 *)(v24 + 16LL * v25);
        v28 = *v27;
        if ( v57 == *v27 )
          goto LABEL_18;
        ++v55;
      }
LABEL_51:
      if ( v49 )
        v27 = v49;
LABEL_18:
      *(_DWORD *)(v7 + 16) = v26;
      if ( *v27 != -8 )
        --*(_DWORD *)(v7 + 20);
      *v27 = v6;
      *((_DWORD *)v27 + 2) = 0;
      v19 = *(_DWORD *)(v7 + 24);
      if ( !v19 )
      {
        ++*(_QWORD *)v7;
        v65 = v21;
        goto LABEL_22;
      }
      v10 = *(_QWORD *)(v7 + 8);
      v9 = v19 - 1;
      v14 = 0;
LABEL_11:
      v15 = ((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4);
      v16 = v15 & v9;
      v17 = (__int64 *)(v10 + 16LL * (v15 & v9));
      v18 = *v17;
      if ( v20 != *v17 )
        break;
LABEL_12:
      --v8;
      if ( *((_DWORD *)v17 + 2) <= v14 )
      {
        v65 = v21;
        goto LABEL_27;
      }
      v8[1] = *v8;
    }
    v56 = 1;
    v59 = 0;
    while ( v18 != -8 )
    {
      if ( !v59 )
      {
        if ( v18 != -16 )
          v17 = 0;
        v59 = v17;
      }
      v16 = v9 & (v56 + v16);
      v17 = (__int64 *)(v10 + 16LL * v16);
      v18 = *v17;
      if ( v20 == *v17 )
        goto LABEL_12;
      ++v56;
    }
    v34 = v59;
    v65 = v21;
    if ( !v59 )
      v34 = v17;
    v36 = *(_DWORD *)(v7 + 16);
    ++*(_QWORD *)v7;
    v33 = v36 + 1;
    if ( 4 * v33 >= 3 * v19 )
    {
LABEL_22:
      v58 = v6;
      sub_1E20790(v7, 2 * v19);
      v29 = *(_DWORD *)(v7 + 24);
      if ( !v29 )
        goto LABEL_93;
      v30 = v29 - 1;
      v31 = *(_QWORD *)(v7 + 8);
      v6 = v58;
      v32 = v30 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v33 = *(_DWORD *)(v7 + 16) + 1;
      v34 = (__int64 *)(v31 + 16LL * v32);
      v35 = *v34;
      if ( v20 != *v34 )
      {
        v53 = 1;
        v54 = 0;
        while ( v35 != -8 )
        {
          if ( !v54 && v35 == -16 )
            v54 = v34;
          v32 = v30 & (v53 + v32);
          v34 = (__int64 *)(v31 + 16LL * v32);
          v35 = *v34;
          if ( v20 == *v34 )
            goto LABEL_24;
          ++v53;
        }
        if ( v54 )
          v34 = v54;
      }
    }
    else if ( v19 - (v33 + *(_DWORD *)(v7 + 20)) <= v19 >> 3 )
    {
      v60 = v6;
      sub_1E20790(v7, v19);
      v37 = *(_DWORD *)(v7 + 24);
      if ( v37 )
      {
        v38 = v37 - 1;
        v39 = *(_QWORD *)(v7 + 8);
        v40 = 0;
        v41 = v38 & v15;
        v6 = v60;
        v42 = 1;
        v33 = *(_DWORD *)(v7 + 16) + 1;
        v34 = (__int64 *)(v39 + 16LL * v41);
        v43 = *v34;
        if ( v20 != *v34 )
        {
          while ( v43 != -8 )
          {
            if ( !v40 && v43 == -16 )
              v40 = v34;
            v41 = v38 & (v42 + v41);
            v34 = (__int64 *)(v39 + 16LL * v41);
            v43 = *v34;
            if ( v20 == *v34 )
              goto LABEL_24;
            ++v42;
          }
          if ( v40 )
            v34 = v40;
        }
        goto LABEL_24;
      }
LABEL_93:
      ++*(_DWORD *)(v7 + 16);
      BUG();
    }
LABEL_24:
    *(_DWORD *)(v7 + 16) = v33;
    if ( *v34 != -8 )
      --*(_DWORD *)(v7 + 20);
    *v34 = v20;
    *((_DWORD *)v34 + 2) = 0;
LABEL_27:
    ++v3;
    *v65 = v6;
  }
  while ( a2 != v3 );
}
