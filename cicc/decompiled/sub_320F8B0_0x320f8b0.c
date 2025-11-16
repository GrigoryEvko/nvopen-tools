// Function: sub_320F8B0
// Address: 0x320f8b0
//
__int64 __fastcall sub_320F8B0(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // r14
  unsigned int v10; // esi
  __int64 v11; // rcx
  int v12; // r11d
  __int64 v13; // r8
  unsigned int v14; // edx
  __int64 v15; // rax
  __int64 v16; // r10
  int v18; // eax
  int v19; // edx
  __int64 v20; // rdx
  int v21; // eax
  __int64 v22; // r14
  __int64 *v23; // rdx
  __int64 v24; // rax
  int v25; // eax
  int v26; // ecx
  __int64 v27; // rsi
  unsigned int v28; // eax
  __int64 v29; // rdi
  int v30; // r10d
  __int64 v31; // rcx
  __int64 *v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rdi
  _QWORD *v35; // rdx
  _QWORD *v36; // rcx
  _QWORD *v37; // rax
  __int64 v38; // rsi
  unsigned __int64 v39; // rdi
  int v40; // r12d
  int v41; // eax
  int v42; // eax
  int v43; // eax
  __int64 v44; // rsi
  unsigned int v45; // r15d
  __int64 v46; // rdi
  __int64 v47; // rcx
  unsigned __int64 v48[7]; // [rsp+8h] [rbp-38h] BYREF

  v9 = *a2;
  v10 = *(_DWORD *)(a1 + 24);
  if ( !v10 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_22;
  }
  v11 = *(_QWORD *)(a1 + 8);
  v12 = 1;
  v13 = 0;
  v14 = (v10 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
  v15 = v11 + 16LL * v14;
  v16 = *(_QWORD *)v15;
  if ( v9 == *(_QWORD *)v15 )
    return *(_QWORD *)(a1 + 32) + 16LL * *(unsigned int *)(v15 + 8);
  while ( v16 != -4096 )
  {
    if ( v16 == -8192 && !v13 )
      v13 = v15;
    a6 = (unsigned int)(v12 + 1);
    v14 = (v10 - 1) & (v12 + v14);
    v15 = v11 + 16LL * v14;
    v16 = *(_QWORD *)v15;
    if ( v9 == *(_QWORD *)v15 )
      return *(_QWORD *)(a1 + 32) + 16LL * *(unsigned int *)(v15 + 8);
    ++v12;
  }
  if ( !v13 )
    v13 = v15;
  v18 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v19 = v18 + 1;
  if ( 4 * (v18 + 1) >= 3 * v10 )
  {
LABEL_22:
    sub_D1FCE0(a1, 2 * v10);
    v25 = *(_DWORD *)(a1 + 24);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(a1 + 8);
      v28 = (v25 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v19 = *(_DWORD *)(a1 + 16) + 1;
      v13 = v27 + 16LL * v28;
      v29 = *(_QWORD *)v13;
      if ( v9 != *(_QWORD *)v13 )
      {
        v30 = 1;
        a6 = 0;
        while ( v29 != -4096 )
        {
          if ( !a6 && v29 == -8192 )
            a6 = v13;
          v28 = v26 & (v30 + v28);
          v13 = v27 + 16LL * v28;
          v29 = *(_QWORD *)v13;
          if ( v9 == *(_QWORD *)v13 )
            goto LABEL_14;
          ++v30;
        }
        if ( a6 )
          v13 = a6;
      }
      goto LABEL_14;
    }
    goto LABEL_57;
  }
  if ( v10 - *(_DWORD *)(a1 + 20) - v19 <= v10 >> 3 )
  {
    sub_D1FCE0(a1, v10);
    v42 = *(_DWORD *)(a1 + 24);
    if ( v42 )
    {
      v43 = v42 - 1;
      v44 = *(_QWORD *)(a1 + 8);
      a6 = 1;
      v45 = v43 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v19 = *(_DWORD *)(a1 + 16) + 1;
      v46 = 0;
      v13 = v44 + 16LL * v45;
      v47 = *(_QWORD *)v13;
      if ( v9 != *(_QWORD *)v13 )
      {
        while ( v47 != -4096 )
        {
          if ( !v46 && v47 == -8192 )
            v46 = v13;
          v45 = v43 & (a6 + v45);
          v13 = v44 + 16LL * v45;
          v47 = *(_QWORD *)v13;
          if ( v9 == *(_QWORD *)v13 )
            goto LABEL_14;
          a6 = (unsigned int)(a6 + 1);
        }
        if ( v46 )
          v13 = v46;
      }
      goto LABEL_14;
    }
LABEL_57:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(a1 + 16) = v19;
  if ( *(_QWORD *)v13 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *(_DWORD *)(v13 + 8) = 0;
  *(_QWORD *)v13 = v9;
  *(_DWORD *)(v13 + 8) = *(_DWORD *)(a1 + 40);
  v20 = *(unsigned int *)(a1 + 40);
  v21 = v20;
  if ( *(_DWORD *)(a1 + 44) <= (unsigned int)v20 )
  {
    v22 = sub_C8D7D0(a1 + 32, a1 + 48, 0, 0x10u, v48, a6);
    v31 = 16LL * *(unsigned int *)(a1 + 40);
    v32 = (__int64 *)(v31 + v22);
    if ( v31 + v22 )
    {
      *v32 = *a2;
      v32[1] = *a3;
      *a3 = 0;
      v31 = 16LL * *(unsigned int *)(a1 + 40);
    }
    v33 = *(_QWORD *)(a1 + 32);
    v34 = v33 + v31;
    if ( v33 == v33 + v31 )
    {
      v38 = v33 + v31;
    }
    else
    {
      v35 = (_QWORD *)(v33 + 8);
      v36 = (_QWORD *)(v22 + v31);
      v37 = (_QWORD *)v22;
      do
      {
        if ( v37 )
        {
          *v37 = *(v35 - 1);
          v37[1] = *v35;
          *v35 = 0;
        }
        v37 += 2;
        v35 += 2;
      }
      while ( v36 != v37 );
      v34 = *(_QWORD *)(a1 + 32);
      v38 = v34 + 16LL * *(unsigned int *)(a1 + 40);
    }
    sub_31F9F70(v34, v38);
    v39 = *(_QWORD *)(a1 + 32);
    v40 = v48[0];
    if ( a1 + 48 != v39 )
      _libc_free(v39);
    v41 = *(_DWORD *)(a1 + 40);
    *(_QWORD *)(a1 + 32) = v22;
    *(_DWORD *)(a1 + 44) = v40;
    v24 = (unsigned int)(v41 + 1);
    *(_DWORD *)(a1 + 40) = v24;
  }
  else
  {
    v22 = *(_QWORD *)(a1 + 32);
    v23 = (__int64 *)(v22 + 16 * v20);
    if ( v23 )
    {
      *v23 = *a2;
      v23[1] = *a3;
      *a3 = 0;
      v21 = *(_DWORD *)(a1 + 40);
      v22 = *(_QWORD *)(a1 + 32);
    }
    v24 = (unsigned int)(v21 + 1);
    *(_DWORD *)(a1 + 40) = v24;
  }
  return v22 + 16 * v24 - 16;
}
