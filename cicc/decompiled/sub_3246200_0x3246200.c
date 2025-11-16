// Function: sub_3246200
// Address: 0x3246200
//
__int64 __fastcall sub_3246200(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rdi
  unsigned int v7; // esi
  __int64 v8; // r9
  int v9; // r10d
  __int64 v10; // r8
  unsigned int v11; // ecx
  __int64 v12; // r14
  _QWORD *v13; // rax
  __int64 v14; // rdx
  __int64 result; // rax
  _QWORD *v16; // rbx
  int v17; // ecx
  int v18; // ecx
  int v19; // eax
  int v20; // esi
  __int64 v21; // rdi
  unsigned int v22; // edx
  __int64 v23; // r8
  int v24; // r10d
  _QWORD *v25; // r9
  int v26; // eax
  int v27; // edx
  __int64 v28; // rdi
  int v29; // r9d
  _QWORD *v30; // r8
  unsigned int v31; // r15d
  __int64 v32; // rsi

  v6 = a1 + 368;
  v7 = *(_DWORD *)(a1 + 392);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 368);
    goto LABEL_20;
  }
  v8 = v7 - 1;
  v9 = 1;
  v10 = *(_QWORD *)(a1 + 376);
  v11 = v8 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v12 = v10 + 56LL * v11;
  v13 = 0;
  v14 = *(_QWORD *)v12;
  if ( *(_QWORD *)v12 == a2 )
  {
LABEL_3:
    result = *(unsigned int *)(v12 + 16);
    v16 = (_QWORD *)(v12 + 8);
    if ( result + 1 > (unsigned __int64)*(unsigned int *)(v12 + 20) )
    {
      sub_C8D5F0(v12 + 8, (const void *)(v12 + 24), result + 1, 8u, v10, v8);
      result = *(unsigned int *)(v12 + 16);
    }
    goto LABEL_5;
  }
  while ( v14 != -4096 )
  {
    if ( !v13 && v14 == -8192 )
      v13 = (_QWORD *)v12;
    v11 = v8 & (v9 + v11);
    v12 = v10 + 56LL * v11;
    v14 = *(_QWORD *)v12;
    if ( *(_QWORD *)v12 == a2 )
      goto LABEL_3;
    ++v9;
  }
  v17 = *(_DWORD *)(a1 + 384);
  if ( !v13 )
    v13 = (_QWORD *)v12;
  ++*(_QWORD *)(a1 + 368);
  v18 = v17 + 1;
  if ( 4 * v18 >= 3 * v7 )
  {
LABEL_20:
    sub_3245EA0(v6, 2 * v7);
    v19 = *(_DWORD *)(a1 + 392);
    if ( v19 )
    {
      v20 = v19 - 1;
      v21 = *(_QWORD *)(a1 + 376);
      v22 = (v19 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v13 = (_QWORD *)(v21 + 56LL * v22);
      v23 = *v13;
      v18 = *(_DWORD *)(a1 + 384) + 1;
      if ( *v13 != a2 )
      {
        v24 = 1;
        v25 = 0;
        while ( v23 != -4096 )
        {
          if ( !v25 && v23 == -8192 )
            v25 = v13;
          v22 = v20 & (v24 + v22);
          v13 = (_QWORD *)(v21 + 56LL * v22);
          v23 = *v13;
          if ( *v13 == a2 )
            goto LABEL_16;
          ++v24;
        }
        if ( v25 )
          v13 = v25;
      }
      goto LABEL_16;
    }
    goto LABEL_43;
  }
  if ( v7 - *(_DWORD *)(a1 + 388) - v18 <= v7 >> 3 )
  {
    sub_3245EA0(v6, v7);
    v26 = *(_DWORD *)(a1 + 392);
    if ( v26 )
    {
      v27 = v26 - 1;
      v28 = *(_QWORD *)(a1 + 376);
      v29 = 1;
      v30 = 0;
      v31 = (v26 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v13 = (_QWORD *)(v28 + 56LL * v31);
      v32 = *v13;
      v18 = *(_DWORD *)(a1 + 384) + 1;
      if ( *v13 != a2 )
      {
        while ( v32 != -4096 )
        {
          if ( v32 == -8192 && !v30 )
            v30 = v13;
          v31 = v27 & (v29 + v31);
          v13 = (_QWORD *)(v28 + 56LL * v31);
          v32 = *v13;
          if ( *v13 == a2 )
            goto LABEL_16;
          ++v29;
        }
        if ( v30 )
          v13 = v30;
      }
      goto LABEL_16;
    }
LABEL_43:
    ++*(_DWORD *)(a1 + 384);
    BUG();
  }
LABEL_16:
  *(_DWORD *)(a1 + 384) = v18;
  if ( *v13 != -4096 )
    --*(_DWORD *)(a1 + 388);
  *v13 = a2;
  v16 = v13 + 1;
  v13[1] = v13 + 3;
  v13[2] = 0x400000000LL;
  result = 0;
LABEL_5:
  *(_QWORD *)(*v16 + 8 * result) = a3;
  ++*((_DWORD *)v16 + 2);
  return result;
}
