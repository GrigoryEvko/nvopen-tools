// Function: sub_22D9C50
// Address: 0x22d9c50
//
_DWORD *__fastcall sub_22D9C50(__int64 a1, __int64 a2)
{
  int v4; // eax
  __int64 v5; // rsi
  int v6; // ecx
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // rdi
  int v10; // r13d
  unsigned int v11; // esi
  __int64 v12; // r9
  __int64 v13; // rdi
  int v14; // r11d
  unsigned int v15; // ecx
  _DWORD *v16; // r8
  _DWORD *v17; // rax
  int v18; // edx
  int v20; // eax
  __int64 v21; // rcx
  int v22; // r13d
  unsigned int v23; // edx
  __int64 *v24; // rax
  __int64 v25; // rsi
  int v26; // eax
  int v27; // esi
  __int64 v28; // r8
  unsigned int v29; // edx
  int v30; // edi
  int v31; // ecx
  int v32; // r10d
  _DWORD *v33; // r9
  int v34; // ecx
  int v35; // r8d
  int v36; // eax
  int v37; // edx
  __int64 v38; // rdi
  int v39; // r9d
  _DWORD *v40; // r8
  unsigned int v41; // r12d
  int v42; // esi
  int v43; // eax
  int v44; // edi
  unsigned __int64 v45[2]; // [rsp+0h] [rbp-70h] BYREF
  _BYTE v46[96]; // [rsp+10h] [rbp-60h] BYREF

  v4 = *(_DWORD *)(a1 + 32);
  v5 = *(_QWORD *)(a1 + 16);
  if ( v4 )
  {
    v6 = v4 - 1;
    v7 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v5 + 16LL * v7);
    v9 = *v8;
    if ( a2 == *v8 )
    {
LABEL_3:
      v10 = *((_DWORD *)v8 + 2);
      if ( v10 )
        goto LABEL_4;
    }
    else
    {
      v20 = 1;
      while ( v9 != -4096 )
      {
        v35 = v20 + 1;
        v7 = v6 & (v20 + v7);
        v8 = (__int64 *)(v5 + 16LL * v7);
        v9 = *v8;
        if ( a2 == *v8 )
          goto LABEL_3;
        v20 = v35;
      }
    }
  }
  v45[0] = (unsigned __int64)v46;
  v45[1] = 0x800000000LL;
  sub_22D81B0((int *)a1, a2, (__int64)v45);
  v10 = *(_DWORD *)(a1 + 32);
  v21 = *(_QWORD *)(a1 + 16);
  if ( v10 )
  {
    v22 = v10 - 1;
    v23 = v22 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v24 = (__int64 *)(v21 + 16LL * v23);
    v25 = *v24;
    if ( a2 == *v24 )
    {
LABEL_11:
      v10 = *((_DWORD *)v24 + 2);
    }
    else
    {
      v43 = 1;
      while ( v25 != -4096 )
      {
        v44 = v43 + 1;
        v23 = v22 & (v43 + v23);
        v24 = (__int64 *)(v21 + 16LL * v23);
        v25 = *v24;
        if ( a2 == *v24 )
          goto LABEL_11;
        v43 = v44;
      }
      v10 = 0;
    }
  }
  if ( (_BYTE *)v45[0] == v46 )
  {
LABEL_4:
    v11 = *(_DWORD *)(a1 + 64);
    v12 = a1 + 40;
    if ( v11 )
      goto LABEL_5;
LABEL_14:
    ++*(_QWORD *)(a1 + 40);
    goto LABEL_15;
  }
  _libc_free(v45[0]);
  v11 = *(_DWORD *)(a1 + 64);
  v12 = a1 + 40;
  if ( !v11 )
    goto LABEL_14;
LABEL_5:
  v13 = *(_QWORD *)(a1 + 48);
  v14 = 1;
  v15 = (v11 - 1) & (37 * v10);
  v16 = (_DWORD *)(v13 + 88LL * v15);
  v17 = 0;
  v18 = *v16;
  if ( *v16 == v10 )
    return v16 + 2;
  while ( v18 != -1 )
  {
    if ( v18 == -2 && !v17 )
      v17 = v16;
    v15 = (v11 - 1) & (v14 + v15);
    v16 = (_DWORD *)(v13 + 88LL * v15);
    v18 = *v16;
    if ( *v16 == v10 )
      return v16 + 2;
    ++v14;
  }
  v34 = *(_DWORD *)(a1 + 56);
  if ( !v17 )
    v17 = v16;
  ++*(_QWORD *)(a1 + 40);
  v31 = v34 + 1;
  if ( 4 * v31 >= 3 * v11 )
  {
LABEL_15:
    sub_22D7160(v12, 2 * v11);
    v26 = *(_DWORD *)(a1 + 64);
    if ( v26 )
    {
      v27 = v26 - 1;
      v28 = *(_QWORD *)(a1 + 48);
      v29 = (v26 - 1) & (37 * v10);
      v17 = (_DWORD *)(v28 + 88LL * v29);
      v30 = *v17;
      v31 = *(_DWORD *)(a1 + 56) + 1;
      if ( *v17 != v10 )
      {
        v32 = 1;
        v33 = 0;
        while ( v30 != -1 )
        {
          if ( !v33 && v30 == -2 )
            v33 = v17;
          v29 = v27 & (v32 + v29);
          v17 = (_DWORD *)(v28 + 88LL * v29);
          v30 = *v17;
          if ( *v17 == v10 )
            goto LABEL_32;
          ++v32;
        }
        if ( v33 )
          v17 = v33;
      }
      goto LABEL_32;
    }
    goto LABEL_58;
  }
  if ( v11 - *(_DWORD *)(a1 + 60) - v31 <= v11 >> 3 )
  {
    sub_22D7160(v12, v11);
    v36 = *(_DWORD *)(a1 + 64);
    if ( v36 )
    {
      v37 = v36 - 1;
      v38 = *(_QWORD *)(a1 + 48);
      v39 = 1;
      v40 = 0;
      v41 = (v36 - 1) & (37 * v10);
      v17 = (_DWORD *)(v38 + 88LL * v41);
      v42 = *v17;
      v31 = *(_DWORD *)(a1 + 56) + 1;
      if ( *v17 != v10 )
      {
        while ( v42 != -1 )
        {
          if ( v42 == -2 && !v40 )
            v40 = v17;
          v41 = v37 & (v39 + v41);
          v17 = (_DWORD *)(v38 + 88LL * v41);
          v42 = *v17;
          if ( *v17 == v10 )
            goto LABEL_32;
          ++v39;
        }
        if ( v40 )
          v17 = v40;
      }
      goto LABEL_32;
    }
LABEL_58:
    ++*(_DWORD *)(a1 + 56);
    BUG();
  }
LABEL_32:
  *(_DWORD *)(a1 + 56) = v31;
  if ( *v17 != -1 )
    --*(_DWORD *)(a1 + 60);
  *v17 = v10;
  *((_QWORD *)v17 + 5) = v17 + 14;
  *((_QWORD *)v17 + 6) = 0x400000000LL;
  *(_OWORD *)(v17 + 2) = 0;
  *(_OWORD *)(v17 + 6) = 0;
  *(_OWORD *)(v17 + 14) = 0;
  *(_OWORD *)(v17 + 18) = 0;
  return v17 + 2;
}
