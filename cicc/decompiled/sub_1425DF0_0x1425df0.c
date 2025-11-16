// Function: sub_1425DF0
// Address: 0x1425df0
//
__int64 __fastcall sub_1425DF0(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  __int64 v5; // rsi
  __int64 v6; // rcx
  int v7; // r11d
  _QWORD *v8; // r13
  unsigned int v9; // edx
  _QWORD *v10; // rax
  __int64 v11; // r9
  __int64 result; // rax
  int v13; // eax
  int v14; // edx
  __int64 v15; // rcx
  __int64 v16; // r8
  unsigned __int64 *v17; // r12
  unsigned __int64 *v18; // rbx
  unsigned __int64 *v19; // rdi
  unsigned __int64 v20; // rdx
  int v21; // eax
  int v22; // ecx
  __int64 v23; // rdi
  unsigned int v24; // eax
  int v25; // r9d
  _QWORD *v26; // r8
  int v27; // eax
  int v28; // eax
  int v29; // r8d
  unsigned int v30; // r14d
  _QWORD *v31; // rdi
  __int64 v32; // rcx

  v4 = a1 + 56;
  v5 = *(unsigned int *)(a1 + 80);
  if ( !(_DWORD)v5 )
  {
    ++*(_QWORD *)(a1 + 56);
    goto LABEL_24;
  }
  v6 = *(_QWORD *)(a1 + 64);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (_QWORD *)(v6 + 16LL * v9);
  v11 = *v10;
  if ( a2 == *v10 )
    return v10[1];
  while ( v11 != -8 )
  {
    if ( !v8 && v11 == -16 )
      v8 = v10;
    v9 = (v5 - 1) & (v7 + v9);
    v10 = (_QWORD *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( a2 == *v10 )
      return v10[1];
    ++v7;
  }
  if ( !v8 )
    v8 = v10;
  v13 = *(_DWORD *)(a1 + 72);
  ++*(_QWORD *)(a1 + 56);
  v14 = v13 + 1;
  if ( 4 * (v13 + 1) >= (unsigned int)(3 * v5) )
  {
LABEL_24:
    sub_1425BA0(v4, 2 * v5);
    v21 = *(_DWORD *)(a1 + 80);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a1 + 64);
      v24 = (v21 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v14 = *(_DWORD *)(a1 + 72) + 1;
      v8 = (_QWORD *)(v23 + 16LL * v24);
      v5 = *v8;
      if ( a2 != *v8 )
      {
        v25 = 1;
        v26 = 0;
        while ( v5 != -8 )
        {
          if ( !v26 && v5 == -16 )
            v26 = v8;
          v24 = v22 & (v25 + v24);
          v8 = (_QWORD *)(v23 + 16LL * v24);
          v5 = *v8;
          if ( a2 == *v8 )
            goto LABEL_15;
          ++v25;
        }
        if ( v26 )
          v8 = v26;
      }
      goto LABEL_15;
    }
    goto LABEL_47;
  }
  if ( (int)v5 - *(_DWORD *)(a1 + 76) - v14 <= (unsigned int)v5 >> 3 )
  {
    sub_1425BA0(v4, v5);
    v27 = *(_DWORD *)(a1 + 80);
    if ( v27 )
    {
      v28 = v27 - 1;
      v5 = *(_QWORD *)(a1 + 64);
      v29 = 1;
      v30 = v28 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v31 = 0;
      v14 = *(_DWORD *)(a1 + 72) + 1;
      v8 = (_QWORD *)(v5 + 16LL * v30);
      v32 = *v8;
      if ( a2 != *v8 )
      {
        while ( v32 != -8 )
        {
          if ( !v31 && v32 == -16 )
            v31 = v8;
          v30 = v28 & (v29 + v30);
          v8 = (_QWORD *)(v5 + 16LL * v30);
          v32 = *v8;
          if ( a2 == *v8 )
            goto LABEL_15;
          ++v29;
        }
        if ( v31 )
          v8 = v31;
      }
      goto LABEL_15;
    }
LABEL_47:
    ++*(_DWORD *)(a1 + 72);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 72) = v14;
  if ( *v8 != -8 )
    --*(_DWORD *)(a1 + 76);
  *v8 = a2;
  v8[1] = 0;
  result = sub_22077B0(16);
  if ( result )
  {
    *(_QWORD *)(result + 8) = result;
    *(_QWORD *)result = result | 4;
  }
  v17 = (unsigned __int64 *)v8[1];
  v8[1] = result;
  if ( v17 )
  {
    v18 = (unsigned __int64 *)v17[1];
    while ( v17 != v18 )
    {
      v19 = v18;
      v18 = (unsigned __int64 *)v18[1];
      v20 = *v19 & 0xFFFFFFFFFFFFFFF8LL;
      *v18 = v20 | *v18 & 7;
      *(_QWORD *)(v20 + 8) = v18;
      *v19 &= 7u;
      v19 -= 4;
      v19[5] = 0;
      sub_164BEC0(v19, v5, v20, v15, v16);
    }
    j_j___libc_free_0(v17, 16);
    return v8[1];
  }
  return result;
}
