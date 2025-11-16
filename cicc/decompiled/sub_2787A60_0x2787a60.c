// Function: sub_2787A60
// Address: 0x2787a60
//
void __fastcall sub_2787A60(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v6; // esi
  __int64 v7; // rdi
  int v8; // r11d
  __int64 v9; // rdx
  unsigned int v10; // ecx
  __int64 v11; // rax
  __int64 v12; // r9
  __int64 v13; // rbx
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  int v16; // eax
  int v17; // ecx
  __int64 v18; // rdx
  int v19; // eax
  __int64 v20; // rdx
  int v21; // eax
  int v22; // eax
  int v23; // eax
  int v24; // esi
  __int64 v25; // r8
  unsigned int v26; // eax
  __int64 v27; // rdi
  int v28; // r10d
  __int64 v29; // r14
  __int64 v30; // rdx
  __int64 v31; // rax
  int v32; // edx
  int v33; // edx
  __int64 v34; // rax
  unsigned __int64 v35; // r12
  __int64 v36; // rdx
  int v37; // ecx
  __int64 v38; // r13
  unsigned __int64 v39; // rdi
  unsigned __int64 v40; // rdi
  int v41; // r13d
  int v42; // eax
  int v43; // eax
  __int64 v44; // rdi
  unsigned int v45; // r14d
  __int64 v46; // r8
  __int64 v47; // rsi
  unsigned __int64 v48[7]; // [rsp+8h] [rbp-38h] BYREF

  v6 = *(_DWORD *)(a1 + 24);
  if ( !v6 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_27;
  }
  v7 = *(_QWORD *)(a1 + 8);
  v8 = 1;
  v9 = 0;
  v10 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v11 = v7 + 16LL * v10;
  v12 = *(_QWORD *)v11;
  if ( a2 == *(_QWORD *)v11 )
  {
LABEL_3:
    v13 = *(_QWORD *)(a1 + 32) + 40LL * *(unsigned int *)(v11 + 8);
    if ( *(_DWORD *)(v13 + 16) > 0x40u )
    {
      v14 = *(_QWORD *)(v13 + 8);
      if ( v14 )
        j_j___libc_free_0_0(v14);
    }
    *(_QWORD *)(v13 + 8) = *(_QWORD *)a3;
    *(_DWORD *)(v13 + 16) = *(_DWORD *)(a3 + 8);
    *(_DWORD *)(a3 + 8) = 0;
    if ( *(_DWORD *)(v13 + 32) > 0x40u )
    {
      v15 = *(_QWORD *)(v13 + 24);
      if ( v15 )
        j_j___libc_free_0_0(v15);
    }
    *(_QWORD *)(v13 + 24) = *(_QWORD *)(a3 + 16);
    *(_DWORD *)(v13 + 32) = *(_DWORD *)(a3 + 24);
    *(_DWORD *)(a3 + 24) = 0;
    return;
  }
  while ( v12 != -4096 )
  {
    if ( v12 == -8192 && !v9 )
      v9 = v11;
    v10 = (v6 - 1) & (v8 + v10);
    v11 = v7 + 16LL * v10;
    v12 = *(_QWORD *)v11;
    if ( *(_QWORD *)v11 == a2 )
      goto LABEL_3;
    ++v8;
  }
  if ( !v9 )
    v9 = v11;
  v16 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v17 = v16 + 1;
  if ( 4 * (v16 + 1) >= 3 * v6 )
  {
LABEL_27:
    sub_9BAAD0(a1, 2 * v6);
    v23 = *(_DWORD *)(a1 + 24);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a1 + 8);
      v26 = (v23 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v17 = *(_DWORD *)(a1 + 16) + 1;
      v9 = v25 + 16LL * v26;
      v27 = *(_QWORD *)v9;
      if ( *(_QWORD *)v9 != a2 )
      {
        v28 = 1;
        v12 = 0;
        while ( v27 != -4096 )
        {
          if ( !v12 && v27 == -8192 )
            v12 = v9;
          v26 = v24 & (v28 + v26);
          v9 = v25 + 16LL * v26;
          v27 = *(_QWORD *)v9;
          if ( *(_QWORD *)v9 == a2 )
            goto LABEL_20;
          ++v28;
        }
        if ( v12 )
          v9 = v12;
      }
      goto LABEL_20;
    }
    goto LABEL_69;
  }
  if ( v6 - *(_DWORD *)(a1 + 20) - v17 <= v6 >> 3 )
  {
    sub_9BAAD0(a1, v6);
    v42 = *(_DWORD *)(a1 + 24);
    if ( v42 )
    {
      v43 = v42 - 1;
      v44 = *(_QWORD *)(a1 + 8);
      v12 = 1;
      v45 = v43 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v46 = 0;
      v17 = *(_DWORD *)(a1 + 16) + 1;
      v9 = v44 + 16LL * v45;
      v47 = *(_QWORD *)v9;
      if ( *(_QWORD *)v9 != a2 )
      {
        while ( v47 != -4096 )
        {
          if ( !v46 && v47 == -8192 )
            v46 = v9;
          v45 = v43 & (v12 + v45);
          v9 = v44 + 16LL * v45;
          v47 = *(_QWORD *)v9;
          if ( *(_QWORD *)v9 == a2 )
            goto LABEL_20;
          v12 = (unsigned int)(v12 + 1);
        }
        if ( v46 )
          v9 = v46;
      }
      goto LABEL_20;
    }
LABEL_69:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_20:
  *(_DWORD *)(a1 + 16) = v17;
  if ( *(_QWORD *)v9 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *(_DWORD *)(v9 + 8) = 0;
  *(_QWORD *)v9 = a2;
  *(_DWORD *)(v9 + 8) = *(_DWORD *)(a1 + 40);
  v18 = *(unsigned int *)(a1 + 40);
  v19 = v18;
  if ( *(_DWORD *)(a1 + 44) <= (unsigned int)v18 )
  {
    v29 = sub_C8D7D0(a1 + 32, a1 + 48, 0, 0x28u, v48, v12);
    v30 = 40LL * *(unsigned int *)(a1 + 40);
    v31 = v30 + v29;
    if ( v30 + v29 )
    {
      *(_QWORD *)v31 = a2;
      v32 = *(_DWORD *)(a3 + 8);
      *(_DWORD *)(a3 + 8) = 0;
      *(_DWORD *)(v31 + 16) = v32;
      *(_QWORD *)(v31 + 8) = *(_QWORD *)a3;
      v33 = *(_DWORD *)(a3 + 24);
      *(_DWORD *)(a3 + 24) = 0;
      *(_DWORD *)(v31 + 32) = v33;
      *(_QWORD *)(v31 + 24) = *(_QWORD *)(a3 + 16);
      v30 = 40LL * *(unsigned int *)(a1 + 40);
    }
    v34 = *(_QWORD *)(a1 + 32);
    v35 = v34 + v30;
    if ( v34 != v34 + v30 )
    {
      v36 = v29;
      do
      {
        if ( v36 )
        {
          *(_QWORD *)v36 = *(_QWORD *)v34;
          *(_DWORD *)(v36 + 16) = *(_DWORD *)(v34 + 16);
          *(_QWORD *)(v36 + 8) = *(_QWORD *)(v34 + 8);
          v37 = *(_DWORD *)(v34 + 32);
          *(_DWORD *)(v34 + 16) = 0;
          *(_DWORD *)(v36 + 32) = v37;
          *(_QWORD *)(v36 + 24) = *(_QWORD *)(v34 + 24);
          *(_DWORD *)(v34 + 32) = 0;
        }
        v34 += 40;
        v36 += 40;
      }
      while ( v35 != v34 );
      v38 = *(_QWORD *)(a1 + 32);
      v35 = v38 + 40LL * *(unsigned int *)(a1 + 40);
      if ( v38 != v35 )
      {
        do
        {
          v35 -= 40LL;
          if ( *(_DWORD *)(v35 + 32) > 0x40u )
          {
            v39 = *(_QWORD *)(v35 + 24);
            if ( v39 )
              j_j___libc_free_0_0(v39);
          }
          if ( *(_DWORD *)(v35 + 16) > 0x40u )
          {
            v40 = *(_QWORD *)(v35 + 8);
            if ( v40 )
              j_j___libc_free_0_0(v40);
          }
        }
        while ( v38 != v35 );
        v35 = *(_QWORD *)(a1 + 32);
      }
    }
    v41 = v48[0];
    if ( v35 != a1 + 48 )
      _libc_free(v35);
    ++*(_DWORD *)(a1 + 40);
    *(_QWORD *)(a1 + 32) = v29;
    *(_DWORD *)(a1 + 44) = v41;
  }
  else
  {
    v20 = *(_QWORD *)(a1 + 32) + 40 * v18;
    if ( v20 )
    {
      *(_QWORD *)v20 = a2;
      v21 = *(_DWORD *)(a3 + 8);
      *(_DWORD *)(a3 + 8) = 0;
      *(_DWORD *)(v20 + 16) = v21;
      *(_QWORD *)(v20 + 8) = *(_QWORD *)a3;
      v22 = *(_DWORD *)(a3 + 24);
      *(_DWORD *)(a3 + 24) = 0;
      *(_DWORD *)(v20 + 32) = v22;
      *(_QWORD *)(v20 + 24) = *(_QWORD *)(a3 + 16);
      v19 = *(_DWORD *)(a1 + 40);
    }
    *(_DWORD *)(a1 + 40) = v19 + 1;
  }
}
