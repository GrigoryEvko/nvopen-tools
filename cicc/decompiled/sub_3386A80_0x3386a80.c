// Function: sub_3386A80
// Address: 0x3386a80
//
__int64 __fastcall sub_3386A80(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // r12
  unsigned int v6; // esi
  __int64 v7; // r8
  __int64 v8; // rdi
  int v9; // r11d
  __int64 v10; // r13
  unsigned int v11; // ecx
  __int64 v12; // rax
  __int64 v13; // r9
  __int64 v14; // rax
  int v16; // eax
  int v17; // ecx
  __int64 v18; // rax
  unsigned __int64 v19; // rcx
  __int64 v20; // rax
  unsigned __int64 v21; // rsi
  int v22; // edi
  __int64 v23; // rcx
  __int64 *v24; // rsi
  __int64 *v25; // rax
  __int64 v26; // rcx
  unsigned __int64 v27; // r14
  __int64 v28; // rcx
  __int64 v29; // r12
  unsigned __int64 v30; // rbx
  __int64 v31; // rcx
  __int64 v32; // rsi
  int v33; // eax
  int v34; // esi
  unsigned int v35; // eax
  __int64 v36; // rdi
  int v37; // r10d
  unsigned __int64 v38; // r12
  __int64 v39; // rdi
  int v40; // eax
  int v41; // eax
  __int64 v42; // rdi
  unsigned int v43; // r14d
  __int64 v44; // rsi
  __int64 v45; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int64 v46; // [rsp+18h] [rbp-48h]
  __int64 v47; // [rsp+20h] [rbp-40h]
  __int64 v48; // [rsp+28h] [rbp-38h]

  v5 = *a2;
  v6 = *(_DWORD *)(a1 + 24);
  if ( !v6 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_29;
  }
  v7 = v6 - 1;
  v8 = *(_QWORD *)(a1 + 8);
  v9 = 1;
  v10 = 0;
  v11 = v7 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v12 = v8 + 16LL * v11;
  v13 = *(_QWORD *)v12;
  if ( v5 == *(_QWORD *)v12 )
  {
LABEL_3:
    v14 = *(unsigned int *)(v12 + 8);
    return *(_QWORD *)(a1 + 32) + 32 * v14 + 8;
  }
  while ( v13 != -4096 )
  {
    if ( !v10 && v13 == -8192 )
      v10 = v12;
    v11 = v7 & (v9 + v11);
    v12 = v8 + 16LL * v11;
    v13 = *(_QWORD *)v12;
    if ( v5 == *(_QWORD *)v12 )
      goto LABEL_3;
    ++v9;
  }
  if ( !v10 )
    v10 = v12;
  v16 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v17 = v16 + 1;
  if ( 4 * (v16 + 1) >= 3 * v6 )
  {
LABEL_29:
    sub_A429D0(a1, 2 * v6);
    v33 = *(_DWORD *)(a1 + 24);
    if ( v33 )
    {
      v34 = v33 - 1;
      v7 = *(_QWORD *)(a1 + 8);
      a3 = *(unsigned int *)(a1 + 16);
      v35 = (v33 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v17 = a3 + 1;
      v10 = v7 + 16LL * v35;
      v36 = *(_QWORD *)v10;
      if ( v5 != *(_QWORD *)v10 )
      {
        v37 = 1;
        v13 = 0;
        while ( v36 != -4096 )
        {
          if ( !v13 && v36 == -8192 )
            v13 = v10;
          v35 = v34 & (v37 + v35);
          v10 = v7 + 16LL * v35;
          v36 = *(_QWORD *)v10;
          if ( v5 == *(_QWORD *)v10 )
            goto LABEL_15;
          ++v37;
        }
        if ( v13 )
          v10 = v13;
      }
      goto LABEL_15;
    }
    goto LABEL_56;
  }
  if ( v6 - *(_DWORD *)(a1 + 20) - v17 <= v6 >> 3 )
  {
    sub_A429D0(a1, v6);
    v40 = *(_DWORD *)(a1 + 24);
    if ( v40 )
    {
      v41 = v40 - 1;
      v42 = *(_QWORD *)(a1 + 8);
      a3 = *(unsigned int *)(a1 + 16);
      v7 = 0;
      v43 = v41 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v13 = 1;
      v17 = a3 + 1;
      v10 = v42 + 16LL * v43;
      v44 = *(_QWORD *)v10;
      if ( v5 != *(_QWORD *)v10 )
      {
        while ( v44 != -4096 )
        {
          if ( !v7 && v44 == -8192 )
            v7 = v10;
          v43 = v41 & (v13 + v43);
          v10 = v42 + 16LL * v43;
          v44 = *(_QWORD *)v10;
          if ( v5 == *(_QWORD *)v10 )
            goto LABEL_15;
          v13 = (unsigned int)(v13 + 1);
        }
        if ( v7 )
          v10 = v7;
      }
      goto LABEL_15;
    }
LABEL_56:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v17;
  if ( *(_QWORD *)v10 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v10 = v5;
  *(_DWORD *)(v10 + 8) = 0;
  v18 = *a2;
  v19 = *(unsigned int *)(a1 + 44);
  v46 = 0;
  v45 = v18;
  v20 = *(unsigned int *)(a1 + 40);
  v47 = 0;
  v21 = v20 + 1;
  v48 = 0;
  v22 = v20;
  if ( v20 + 1 > v19 )
  {
    v38 = *(_QWORD *)(a1 + 32);
    v39 = a1 + 32;
    if ( v38 > (unsigned __int64)&v45 || (unsigned __int64)&v45 >= v38 + 32 * v20 )
    {
      sub_3382E20(v39, v21, a3, v19, v7, v13);
      v20 = *(unsigned int *)(a1 + 40);
      v23 = *(_QWORD *)(a1 + 32);
      v24 = &v45;
      v22 = *(_DWORD *)(a1 + 40);
    }
    else
    {
      sub_3382E20(v39, v21, a3, v19, v7, v13);
      v23 = *(_QWORD *)(a1 + 32);
      v20 = *(unsigned int *)(a1 + 40);
      v24 = (__int64 *)((char *)&v45 + v23 - v38);
      v22 = *(_DWORD *)(a1 + 40);
    }
  }
  else
  {
    v23 = *(_QWORD *)(a1 + 32);
    v24 = &v45;
  }
  v25 = (__int64 *)(v23 + 32 * v20);
  if ( v25 )
  {
    *v25 = *v24;
    v26 = v24[1];
    v24[1] = 0;
    v27 = v46;
    v25[1] = v26;
    v28 = v24[2];
    v24[2] = 0;
    v29 = v47;
    v30 = v27;
    v25[2] = v28;
    v31 = v24[3];
    v24[3] = 0;
    v25[3] = v31;
    ++*(_DWORD *)(a1 + 40);
    if ( v29 != v27 )
    {
      do
      {
        v32 = *(_QWORD *)(v30 + 24);
        if ( v32 )
          sub_B91220(v30 + 24, v32);
        v30 += 32LL;
      }
      while ( v29 != v30 );
    }
    if ( v27 )
      j_j___libc_free_0(v27);
  }
  else
  {
    *(_DWORD *)(a1 + 40) = v22 + 1;
  }
  v14 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
  *(_DWORD *)(v10 + 8) = v14;
  return *(_QWORD *)(a1 + 32) + 32 * v14 + 8;
}
