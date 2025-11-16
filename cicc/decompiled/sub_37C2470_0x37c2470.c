// Function: sub_37C2470
// Address: 0x37c2470
//
__int64 __fastcall sub_37C2470(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r9
  __int64 v6; // r15
  __int64 v7; // r12
  __int64 v8; // r13
  __int64 *v9; // rbx
  __int64 v10; // rsi
  __int64 *v11; // r15
  __int64 v12; // rbx
  __int64 v13; // r14
  __int64 v14; // r12
  __int64 v15; // r9
  unsigned int v16; // r11d
  __int64 v17; // r10
  unsigned int v18; // ecx
  __int64 v19; // rax
  __int64 v20; // r13
  unsigned int v21; // ecx
  __int64 v22; // rdi
  unsigned int v23; // edx
  __int64 v24; // rax
  __int64 v25; // r13
  __int64 *v26; // rdx
  unsigned int v27; // esi
  __int64 v28; // rdx
  __int64 v29; // rdi
  __int64 v30; // rbx
  __int64 v31; // rax
  __int64 v32; // r12
  __int64 v33; // r9
  int v34; // eax
  int v35; // eax
  int v37; // eax
  __int64 v39; // [rsp+10h] [rbp-90h]
  __int64 v40; // [rsp+18h] [rbp-88h]
  __int64 v42; // [rsp+30h] [rbp-70h]
  int v43; // [rsp+30h] [rbp-70h]
  int v44; // [rsp+30h] [rbp-70h]
  __int64 v45; // [rsp+30h] [rbp-70h]
  __int64 v46; // [rsp+38h] [rbp-68h]
  __int64 v47; // [rsp+38h] [rbp-68h]
  __int64 v48; // [rsp+38h] [rbp-68h]
  __int64 v49; // [rsp+38h] [rbp-68h]
  __int64 v50; // [rsp+38h] [rbp-68h]
  __int64 v51; // [rsp+38h] [rbp-68h]
  __int64 v53; // [rsp+40h] [rbp-60h]
  unsigned int v54; // [rsp+48h] [rbp-58h]
  __int64 v55; // [rsp+48h] [rbp-58h]
  __int64 v56; // [rsp+58h] [rbp-48h] BYREF
  __int64 v57; // [rsp+60h] [rbp-40h] BYREF
  __int64 v58[7]; // [rsp+68h] [rbp-38h] BYREF

  v5 = a1;
  v46 = (a3 - 1) / 2;
  v40 = a3 & 1;
  if ( a2 >= v46 )
  {
    v12 = a2;
    v11 = (__int64 *)(a1 + 8 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_32;
  }
  else
  {
    v39 = a5;
    v6 = a2;
    v7 = a5 + 664;
    while ( 1 )
    {
      v8 = 2 * (v6 + 1);
      v9 = (__int64 *)(a1 + 16 * (v6 + 1));
      v10 = *v9;
      v58[0] = *(v9 - 1);
      v57 = v10;
      v54 = *(_DWORD *)sub_2E51790(v7, &v57);
      if ( v54 < *(_DWORD *)sub_2E51790(v7, v58) )
      {
        --v8;
        v9 = (__int64 *)(a1 + 8 * v8);
      }
      *(_QWORD *)(a1 + 8 * v6) = *v9;
      if ( v8 >= v46 )
        break;
      v6 = v8;
    }
    v11 = v9;
    a5 = v39;
    v12 = v8;
    v5 = a1;
    if ( v40 )
      goto LABEL_8;
  }
  if ( (a3 - 2) / 2 == v12 )
  {
    v30 = 2 * v12 + 2;
    v31 = *(_QWORD *)(v5 + 8 * v30 - 8);
    v12 = v30 - 1;
    *v11 = v31;
    v11 = (__int64 *)(v5 + 8 * v12);
  }
LABEL_8:
  if ( v12 <= a2 )
    goto LABEL_32;
  v13 = v5;
  v14 = (v12 - 1) / 2;
  v15 = a4;
  v55 = a5 + 664;
  while ( 1 )
  {
    v11 = (__int64 *)(v13 + 8 * v14);
    v27 = *(_DWORD *)(a5 + 688);
    v57 = v15;
    v28 = *v11;
    v56 = *v11;
    if ( v27 )
    {
      v16 = v27 - 1;
      v17 = *(_QWORD *)(a5 + 672);
      v18 = (v27 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
      v19 = v17 + 16LL * v18;
      v20 = *(_QWORD *)v19;
      if ( v28 == *(_QWORD *)v19 )
      {
LABEL_11:
        v21 = *(_DWORD *)(v19 + 8);
        v22 = v15;
        goto LABEL_12;
      }
      v44 = 1;
      v29 = 0;
      while ( v20 != -4096 )
      {
        if ( v20 == -8192 && !v29 )
          v29 = v19;
        v18 = v16 & (v44 + v18);
        v19 = v17 + 16LL * v18;
        v20 = *(_QWORD *)v19;
        if ( v28 == *(_QWORD *)v19 )
          goto LABEL_11;
        ++v44;
      }
      if ( !v29 )
        v29 = v19;
      v37 = *(_DWORD *)(a5 + 680);
      ++*(_QWORD *)(a5 + 664);
      v58[0] = v29;
      if ( 4 * (v37 + 1) < 3 * v27 )
      {
        if ( v27 - *(_DWORD *)(a5 + 684) - (v37 + 1) <= v27 >> 3 )
        {
          v45 = a5;
          v50 = v15;
          sub_2E515B0(v55, v27);
          sub_2E50510(v55, &v56, v58);
          v28 = v56;
          v29 = v58[0];
          a5 = v45;
          v15 = v50;
        }
        goto LABEL_40;
      }
    }
    else
    {
      ++*(_QWORD *)(a5 + 664);
      v58[0] = 0;
    }
    v42 = a5;
    v47 = v15;
    sub_2E515B0(v55, 2 * v27);
    sub_2E50510(v55, &v56, v58);
    v28 = v56;
    v29 = v58[0];
    v15 = v47;
    a5 = v42;
LABEL_40:
    ++*(_DWORD *)(a5 + 680);
    if ( *(_QWORD *)v29 != -4096 )
      --*(_DWORD *)(a5 + 684);
    *(_QWORD *)v29 = v28;
    *(_DWORD *)(v29 + 8) = 0;
    v27 = *(_DWORD *)(a5 + 688);
    if ( !v27 )
    {
      ++*(_QWORD *)(a5 + 664);
      v33 = v13;
      v58[0] = 0;
LABEL_44:
      v49 = a5;
      v53 = v33;
      sub_2E515B0(v55, 2 * v27);
      sub_2E50510(v55, &v57, v58);
      v22 = v57;
      v32 = v58[0];
      v33 = v53;
      a5 = v49;
      goto LABEL_29;
    }
    v17 = *(_QWORD *)(a5 + 672);
    v22 = v57;
    v16 = v27 - 1;
    v21 = 0;
LABEL_12:
    v23 = v16 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
    v24 = v17 + 16LL * v23;
    v25 = *(_QWORD *)v24;
    if ( *(_QWORD *)v24 != v22 )
      break;
LABEL_13:
    v26 = (__int64 *)(v13 + 8 * v12);
    if ( v21 >= *(_DWORD *)(v24 + 8) )
    {
      v11 = (__int64 *)(v13 + 8 * v12);
      goto LABEL_32;
    }
    v12 = v14;
    *v26 = *v11;
    if ( a2 >= v14 )
      goto LABEL_32;
    v14 = (v14 - 1) / 2;
  }
  v43 = 1;
  v48 = 0;
  while ( v25 != -4096 )
  {
    if ( v25 == -8192 )
    {
      if ( v48 )
        v24 = v48;
      v48 = v24;
    }
    v23 = v16 & (v43 + v23);
    v24 = v17 + 16LL * v23;
    v25 = *(_QWORD *)v24;
    if ( *(_QWORD *)v24 == v22 )
      goto LABEL_13;
    ++v43;
  }
  v32 = v48;
  v33 = v13;
  if ( !v48 )
    v32 = v24;
  v34 = *(_DWORD *)(a5 + 680);
  ++*(_QWORD *)(a5 + 664);
  v35 = v34 + 1;
  v58[0] = v32;
  if ( 4 * v35 >= 3 * v27 )
    goto LABEL_44;
  if ( v27 - (*(_DWORD *)(a5 + 684) + v35) <= v27 >> 3 )
  {
    v51 = a5;
    sub_2E515B0(v55, v27);
    sub_2E50510(v55, &v57, v58);
    v22 = v57;
    v32 = v58[0];
    a5 = v51;
    v33 = v13;
  }
LABEL_29:
  ++*(_DWORD *)(a5 + 680);
  if ( *(_QWORD *)v32 != -4096 )
    --*(_DWORD *)(a5 + 684);
  *(_QWORD *)v32 = v22;
  v11 = (__int64 *)(v33 + 8 * v12);
  *(_DWORD *)(v32 + 8) = 0;
LABEL_32:
  *v11 = a4;
  return a4;
}
