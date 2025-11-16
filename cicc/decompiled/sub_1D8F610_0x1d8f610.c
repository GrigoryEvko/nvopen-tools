// Function: sub_1D8F610
// Address: 0x1d8f610
//
__int64 __fastcall sub_1D8F610(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r8
  __int64 result; // rax
  int v10; // edx
  __int64 v11; // rax
  __int64 v12; // r14
  _QWORD *v13; // rax
  _QWORD *v14; // r13
  char *v15; // rsi
  unsigned int v16; // esi
  __int64 v17; // rdi
  __int64 v18; // r9
  unsigned int v19; // r8d
  __int64 *v20; // rdx
  __int64 v21; // rcx
  int v22; // r9d
  int v23; // r14d
  __int64 *v24; // r11
  int v25; // ecx
  int v26; // r8d
  int v27; // edx
  int v28; // esi
  __int64 v29; // r9
  unsigned int v30; // ecx
  __int64 v31; // rdi
  int v32; // r11d
  __int64 *v33; // r10
  int v34; // edx
  int v35; // ecx
  __int64 v36; // rdi
  __int64 *v37; // r9
  unsigned int v38; // r13d
  int v39; // r10d
  __int64 v40; // rsi
  __int64 v41; // [rsp+8h] [rbp-48h]
  __int64 v42; // [rsp+8h] [rbp-48h]
  __int64 v43[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = *(unsigned int *)(a1 + 264);
  if ( (_DWORD)v4 )
  {
    v5 = *(_QWORD *)(a1 + 248);
    v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
    {
LABEL_3:
      if ( v7 != (__int64 *)(v5 + 16 * v4) )
        return v7[1];
    }
    else
    {
      v10 = 1;
      while ( v8 != -8 )
      {
        v22 = v10 + 1;
        v6 = (v4 - 1) & (v10 + v6);
        v7 = (__int64 *)(v5 + 16LL * v6);
        v8 = *v7;
        if ( a2 == *v7 )
          goto LABEL_3;
        v10 = v22;
      }
    }
  }
  v11 = sub_15E0FA0(a2);
  v12 = sub_1D8EC30(a1, *(void **)v11, *(_QWORD *)(v11 + 8));
  v13 = (_QWORD *)sub_22077B0(72);
  v14 = v13;
  if ( v13 )
    sub_1D8E290(v13, a2, v12);
  v43[0] = (__int64)v14;
  v15 = *(char **)(a1 + 224);
  if ( v15 == *(char **)(a1 + 232) )
  {
    sub_1D8F290((__int64 *)(a1 + 216), v15, v43);
    v14 = (_QWORD *)v43[0];
  }
  else
  {
    if ( v15 )
    {
      *(_QWORD *)v15 = v14;
      *(_QWORD *)(a1 + 224) += 8LL;
      goto LABEL_12;
    }
    *(_QWORD *)(a1 + 224) = 8;
  }
  if ( v14 )
  {
    sub_1D8E2D0(v14);
    j_j___libc_free_0(v14, 72);
  }
LABEL_12:
  v16 = *(_DWORD *)(a1 + 264);
  v17 = a1 + 240;
  result = *(_QWORD *)(*(_QWORD *)(a1 + 224) - 8LL);
  if ( !v16 )
  {
    ++*(_QWORD *)(a1 + 240);
    goto LABEL_31;
  }
  v18 = *(_QWORD *)(a1 + 248);
  v19 = (v16 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v20 = (__int64 *)(v18 + 16LL * v19);
  v21 = *v20;
  if ( a2 != *v20 )
  {
    v23 = 1;
    v24 = 0;
    while ( v21 != -8 )
    {
      if ( !v24 && v21 == -16 )
        v24 = v20;
      v19 = (v16 - 1) & (v23 + v19);
      v20 = (__int64 *)(v18 + 16LL * v19);
      v21 = *v20;
      if ( a2 == *v20 )
        goto LABEL_14;
      ++v23;
    }
    v25 = *(_DWORD *)(a1 + 256);
    if ( v24 )
      v20 = v24;
    ++*(_QWORD *)(a1 + 240);
    v26 = v25 + 1;
    if ( 4 * (v25 + 1) < 3 * v16 )
    {
      if ( v16 - *(_DWORD *)(a1 + 260) - v26 > v16 >> 3 )
      {
LABEL_27:
        *(_DWORD *)(a1 + 256) = v26;
        if ( *v20 != -8 )
          --*(_DWORD *)(a1 + 260);
        *v20 = a2;
        v20[1] = 0;
        goto LABEL_14;
      }
      v42 = result;
      sub_1D8F450(v17, v16);
      v34 = *(_DWORD *)(a1 + 264);
      if ( v34 )
      {
        v35 = v34 - 1;
        v36 = *(_QWORD *)(a1 + 248);
        v37 = 0;
        v38 = (v34 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v39 = 1;
        v26 = *(_DWORD *)(a1 + 256) + 1;
        result = v42;
        v20 = (__int64 *)(v36 + 16LL * v38);
        v40 = *v20;
        if ( a2 != *v20 )
        {
          while ( v40 != -8 )
          {
            if ( !v37 && v40 == -16 )
              v37 = v20;
            v38 = v35 & (v39 + v38);
            v20 = (__int64 *)(v36 + 16LL * v38);
            v40 = *v20;
            if ( a2 == *v20 )
              goto LABEL_27;
            ++v39;
          }
          if ( v37 )
            v20 = v37;
        }
        goto LABEL_27;
      }
LABEL_59:
      ++*(_DWORD *)(a1 + 256);
      BUG();
    }
LABEL_31:
    v41 = result;
    sub_1D8F450(v17, 2 * v16);
    v27 = *(_DWORD *)(a1 + 264);
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *(_QWORD *)(a1 + 248);
      v30 = (v27 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v26 = *(_DWORD *)(a1 + 256) + 1;
      result = v41;
      v20 = (__int64 *)(v29 + 16LL * v30);
      v31 = *v20;
      if ( a2 != *v20 )
      {
        v32 = 1;
        v33 = 0;
        while ( v31 != -8 )
        {
          if ( v31 == -16 && !v33 )
            v33 = v20;
          v30 = v28 & (v32 + v30);
          v20 = (__int64 *)(v29 + 16LL * v30);
          v31 = *v20;
          if ( a2 == *v20 )
            goto LABEL_27;
          ++v32;
        }
        if ( v33 )
          v20 = v33;
      }
      goto LABEL_27;
    }
    goto LABEL_59;
  }
LABEL_14:
  v20[1] = result;
  return result;
}
