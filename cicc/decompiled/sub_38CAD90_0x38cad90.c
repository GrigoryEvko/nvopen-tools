// Function: sub_38CAD90
// Address: 0x38cad90
//
__int64 __fastcall sub_38CAD90(__int64 a1, __int64 *a2)
{
  __int64 v4; // r14
  unsigned int v5; // esi
  __int64 v6; // rcx
  int v7; // r10d
  __int64 *v8; // r13
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r8
  __int64 v12; // rax
  int v14; // eax
  int v15; // edx
  __int64 v16; // rax
  __int64 v17; // rsi
  int v18; // eax
  int v19; // ecx
  __int64 v20; // rdi
  unsigned int v21; // eax
  __int64 v22; // rsi
  int v23; // r9d
  __int64 *v24; // r8
  int v25; // eax
  int v26; // eax
  __int64 v27; // rsi
  int v28; // r8d
  unsigned int v29; // r15d
  __int64 *v30; // rdi
  __int64 v31; // rcx
  __int64 v32; // [rsp+0h] [rbp-50h] BYREF
  unsigned __int64 v33; // [rsp+8h] [rbp-48h]
  __int64 v34; // [rsp+10h] [rbp-40h]
  __int64 v35; // [rsp+18h] [rbp-38h]

  v4 = *a2;
  v5 = *(_DWORD *)(a1 + 24);
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_25;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v10 = (__int64 *)(v6 + 16LL * v9);
  v11 = *v10;
  if ( v4 == *v10 )
  {
LABEL_3:
    v12 = *((unsigned int *)v10 + 2);
    return *(_QWORD *)(a1 + 32) + 32 * v12 + 8;
  }
  while ( v11 != -8 )
  {
    if ( v11 == -16 && !v8 )
      v8 = v10;
    v9 = (v5 - 1) & (v7 + v9);
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( v4 == *v10 )
      goto LABEL_3;
    ++v7;
  }
  if ( !v8 )
    v8 = v10;
  v14 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v15 = v14 + 1;
  if ( 4 * (v14 + 1) >= 3 * v5 )
  {
LABEL_25:
    sub_38CABD0(a1, 2 * v5);
    v18 = *(_DWORD *)(a1 + 24);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(a1 + 8);
      v21 = (v18 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v15 = *(_DWORD *)(a1 + 16) + 1;
      v8 = (__int64 *)(v20 + 16LL * v21);
      v22 = *v8;
      if ( v4 != *v8 )
      {
        v23 = 1;
        v24 = 0;
        while ( v22 != -8 )
        {
          if ( !v24 && v22 == -16 )
            v24 = v8;
          v21 = v19 & (v23 + v21);
          v8 = (__int64 *)(v20 + 16LL * v21);
          v22 = *v8;
          if ( v4 == *v8 )
            goto LABEL_15;
          ++v23;
        }
        if ( v24 )
          v8 = v24;
      }
      goto LABEL_15;
    }
    goto LABEL_48;
  }
  if ( v5 - *(_DWORD *)(a1 + 20) - v15 <= v5 >> 3 )
  {
    sub_38CABD0(a1, v5);
    v25 = *(_DWORD *)(a1 + 24);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(a1 + 8);
      v28 = 1;
      v29 = v26 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v30 = 0;
      v15 = *(_DWORD *)(a1 + 16) + 1;
      v8 = (__int64 *)(v27 + 16LL * v29);
      v31 = *v8;
      if ( v4 != *v8 )
      {
        while ( v31 != -8 )
        {
          if ( !v30 && v31 == -16 )
            v30 = v8;
          v29 = v26 & (v28 + v29);
          v8 = (__int64 *)(v27 + 16LL * v29);
          v31 = *v8;
          if ( v4 == *v8 )
            goto LABEL_15;
          ++v28;
        }
        if ( v30 )
          v8 = v30;
      }
      goto LABEL_15;
    }
LABEL_48:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v15;
  if ( *v8 != -8 )
    --*(_DWORD *)(a1 + 20);
  *v8 = v4;
  *((_DWORD *)v8 + 2) = 0;
  v16 = *a2;
  v33 = 0;
  v17 = *(_QWORD *)(a1 + 40);
  v32 = v16;
  v34 = 0;
  v35 = 0;
  if ( v17 == *(_QWORD *)(a1 + 48) )
  {
    sub_38C9990((unsigned __int64 *)(a1 + 32), (char *)v17, &v32);
    if ( v33 )
      j_j___libc_free_0(v33);
  }
  else
  {
    if ( v17 )
    {
      *(_QWORD *)v17 = v16;
      *(_QWORD *)(v17 + 8) = v33;
      *(_QWORD *)(v17 + 16) = v34;
      *(_QWORD *)(v17 + 24) = v35;
      v17 = *(_QWORD *)(a1 + 40);
    }
    *(_QWORD *)(a1 + 40) = v17 + 32;
  }
  v12 = (unsigned int)((__int64)(*(_QWORD *)(a1 + 40) - *(_QWORD *)(a1 + 32)) >> 5) - 1;
  *((_DWORD *)v8 + 2) = v12;
  return *(_QWORD *)(a1 + 32) + 32 * v12 + 8;
}
