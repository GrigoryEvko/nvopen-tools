// Function: sub_1BCE8C0
// Address: 0x1bce8c0
//
__int64 __fastcall sub_1BCE8C0(__int64 a1, __int64 *a2)
{
  __int64 v4; // r13
  unsigned int v5; // esi
  __int64 v6; // rcx
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r8
  __int64 v10; // rax
  int v12; // r10d
  int v13; // edx
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // r13
  unsigned __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // r12
  __int64 v20; // r15
  __int64 *v21; // r12
  __int64 v22; // r15
  __int64 v23; // r14
  unsigned __int64 v24; // rdi
  int v25; // eax
  int v26; // ecx
  __int64 v27; // rdi
  unsigned int v28; // eax
  __int64 v29; // rsi
  int v30; // r9d
  __int64 *v31; // r8
  int v32; // eax
  int v33; // eax
  __int64 v34; // rsi
  int v35; // r8d
  __int64 *v36; // rdi
  unsigned int v37; // r15d
  __int64 v38; // rcx
  __int64 *v39; // [rsp+0h] [rbp-50h]
  __int64 *v40; // [rsp+0h] [rbp-50h]
  __int64 *v41; // [rsp+8h] [rbp-48h]
  __int64 v42; // [rsp+10h] [rbp-40h] BYREF
  __int64 v43; // [rsp+18h] [rbp-38h]

  v4 = *a2;
  v5 = *(_DWORD *)(a1 + 24);
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_41;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = (v5 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v8 = (__int64 *)(v6 + 16LL * v7);
  v9 = *v8;
  if ( v4 == *v8 )
  {
LABEL_3:
    v10 = *((unsigned int *)v8 + 2);
    return *(_QWORD *)(a1 + 32) + 16 * v10 + 8;
  }
  v39 = 0;
  v12 = 1;
  while ( v9 != -8 )
  {
    if ( !v39 )
    {
      if ( v9 != -16 )
        v8 = 0;
      v39 = v8;
    }
    v7 = (v5 - 1) & (v12 + v7);
    v8 = (__int64 *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( v4 == *v8 )
      goto LABEL_3;
    ++v12;
  }
  if ( v39 )
    v8 = v39;
  ++*(_QWORD *)a1;
  v40 = v8;
  v13 = *(_DWORD *)(a1 + 16) + 1;
  if ( 4 * v13 >= 3 * v5 )
  {
LABEL_41:
    sub_13FEAC0(a1, 2 * v5);
    v25 = *(_DWORD *)(a1 + 24);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(a1 + 8);
      v28 = (v25 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v40 = (__int64 *)(v27 + 16LL * v28);
      v29 = *v40;
      v13 = *(_DWORD *)(a1 + 16) + 1;
      if ( v4 != *v40 )
      {
        v30 = 1;
        v31 = 0;
        while ( v29 != -8 )
        {
          if ( !v31 && v29 == -16 )
            v31 = v40;
          v28 = v26 & (v30 + v28);
          v40 = (__int64 *)(v27 + 16LL * v28);
          v29 = *v40;
          if ( v4 == *v40 )
            goto LABEL_11;
          ++v30;
        }
        if ( !v31 )
          v31 = v40;
        v40 = v31;
      }
      goto LABEL_11;
    }
    goto LABEL_70;
  }
  if ( v5 - *(_DWORD *)(a1 + 20) - v13 <= v5 >> 3 )
  {
    sub_13FEAC0(a1, v5);
    v32 = *(_DWORD *)(a1 + 24);
    if ( v32 )
    {
      v33 = v32 - 1;
      v34 = *(_QWORD *)(a1 + 8);
      v35 = 1;
      v36 = 0;
      v37 = v33 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v40 = (__int64 *)(v34 + 16LL * v37);
      v38 = *v40;
      v13 = *(_DWORD *)(a1 + 16) + 1;
      if ( v4 != *v40 )
      {
        while ( v38 != -8 )
        {
          if ( !v36 && v38 == -16 )
            v36 = v40;
          v37 = v33 & (v35 + v37);
          v40 = (__int64 *)(v34 + 16LL * v37);
          v38 = *v40;
          if ( v4 == *v40 )
            goto LABEL_11;
          ++v35;
        }
        if ( !v36 )
          v36 = v40;
        v40 = v36;
      }
      goto LABEL_11;
    }
LABEL_70:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_11:
  *(_DWORD *)(a1 + 16) = v13;
  if ( *v40 != -8 )
    --*(_DWORD *)(a1 + 20);
  *v40 = v4;
  *((_DWORD *)v40 + 2) = 0;
  v14 = *a2;
  v43 = 0;
  v15 = *(_QWORD *)(a1 + 40);
  v42 = v14;
  if ( v15 == *(_QWORD *)(a1 + 48) )
  {
    sub_1BC2E70((__int64 *)(a1 + 32), (char *)v15, &v42);
    v16 = v43;
    if ( v43 )
    {
      v17 = *(_QWORD *)(v43 + 104);
      if ( v17 != v43 + 120 )
        _libc_free(v17);
      v18 = *(unsigned int *)(v16 + 96);
      if ( (_DWORD)v18 )
      {
        v19 = *(_QWORD *)(v16 + 80);
        v20 = v19 + 88 * v18;
        do
        {
          if ( *(_QWORD *)v19 != -16 && *(_QWORD *)v19 != -8 && (*(_BYTE *)(v19 + 16) & 1) == 0 )
            j___libc_free_0(*(_QWORD *)(v19 + 24));
          v19 += 88;
        }
        while ( v20 != v19 );
      }
      j___libc_free_0(*(_QWORD *)(v16 + 80));
      j___libc_free_0(*(_QWORD *)(v16 + 48));
      v21 = *(__int64 **)(v16 + 8);
      v41 = *(__int64 **)(v16 + 16);
      if ( v41 != v21 )
      {
        do
        {
          v22 = *v21;
          if ( *v21 )
          {
            v23 = v22 + 112LL * *(_QWORD *)(v22 - 8);
            while ( v22 != v23 )
            {
              v23 -= 112;
              v24 = *(_QWORD *)(v23 + 32);
              if ( v24 != v23 + 48 )
                _libc_free(v24);
            }
            j_j_j___libc_free_0_0(v22 - 8);
          }
          ++v21;
        }
        while ( v41 != v21 );
        v21 = *(__int64 **)(v16 + 8);
      }
      if ( v21 )
        j_j___libc_free_0(v21, *(_QWORD *)(v16 + 24) - (_QWORD)v21);
      j_j___libc_free_0(v16, 232);
    }
  }
  else
  {
    if ( v15 )
    {
      *(_QWORD *)v15 = v14;
      *(_QWORD *)(v15 + 8) = v43;
      v15 = *(_QWORD *)(a1 + 40);
    }
    *(_QWORD *)(a1 + 40) = v15 + 16;
  }
  v10 = (unsigned int)((__int64)(*(_QWORD *)(a1 + 40) - *(_QWORD *)(a1 + 32)) >> 4) - 1;
  *((_DWORD *)v40 + 2) = v10;
  return *(_QWORD *)(a1 + 32) + 16 * v10 + 8;
}
