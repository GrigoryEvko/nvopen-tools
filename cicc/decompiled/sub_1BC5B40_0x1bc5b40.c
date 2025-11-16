// Function: sub_1BC5B40
// Address: 0x1bc5b40
//
__int64 __fastcall sub_1BC5B40(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, unsigned int a6)
{
  __int64 v8; // r13
  unsigned int v9; // esi
  __int64 v10; // rcx
  int v11; // r10d
  __int64 v12; // r14
  unsigned int v13; // edx
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // rax
  int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rax
  int v22; // eax
  __int64 v23; // rdi
  unsigned int v24; // eax
  __int64 v25; // rsi
  int v26; // eax
  int v27; // eax
  __int64 v28; // rsi
  unsigned int v29; // r15d
  __int64 v30; // rdi
  __int64 v31; // [rsp+0h] [rbp-60h] BYREF
  _BYTE *v32; // [rsp+8h] [rbp-58h]
  __int64 v33; // [rsp+10h] [rbp-50h]
  _BYTE v34[72]; // [rsp+18h] [rbp-48h] BYREF

  v8 = *a2;
  v9 = *(_DWORD *)(a1 + 24);
  if ( !v9 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_21;
  }
  v10 = *(_QWORD *)(a1 + 8);
  v11 = 1;
  v12 = 0;
  v13 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v14 = v10 + 16LL * v13;
  v15 = *(_QWORD *)v14;
  if ( v8 == *(_QWORD *)v14 )
  {
LABEL_3:
    v16 = *(unsigned int *)(v14 + 8);
    return *(_QWORD *)(a1 + 32) + 40 * v16 + 8;
  }
  while ( v15 != -8 )
  {
    if ( !v12 && v15 == -16 )
      v12 = v14;
    a6 = v11 + 1;
    v13 = (v9 - 1) & (v11 + v13);
    v14 = v10 + 16LL * v13;
    v15 = *(_QWORD *)v14;
    if ( v8 == *(_QWORD *)v14 )
      goto LABEL_3;
    ++v11;
  }
  if ( !v12 )
    v12 = v14;
  v18 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v19 = (unsigned int)(v18 + 1);
  if ( 4 * (int)v19 >= 3 * v9 )
  {
LABEL_21:
    sub_177C7D0(a1, 2 * v9);
    v22 = *(_DWORD *)(a1 + 24);
    if ( v22 )
    {
      v20 = (unsigned int)(v22 - 1);
      v23 = *(_QWORD *)(a1 + 8);
      v24 = v20 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v19 = (unsigned int)(*(_DWORD *)(a1 + 16) + 1);
      v12 = v23 + 16LL * v24;
      v25 = *(_QWORD *)v12;
      if ( v8 != *(_QWORD *)v12 )
      {
        a6 = 1;
        v15 = 0;
        while ( v25 != -8 )
        {
          if ( !v15 && v25 == -16 )
            v15 = v12;
          v24 = v20 & (a6 + v24);
          v12 = v23 + 16LL * v24;
          v25 = *(_QWORD *)v12;
          if ( v8 == *(_QWORD *)v12 )
            goto LABEL_15;
          ++a6;
        }
        if ( v15 )
          v12 = v15;
      }
      goto LABEL_15;
    }
    goto LABEL_44;
  }
  v20 = v9 >> 3;
  if ( v9 - *(_DWORD *)(a1 + 20) - (unsigned int)v19 <= (unsigned int)v20 )
  {
    sub_177C7D0(a1, v9);
    v26 = *(_DWORD *)(a1 + 24);
    if ( v26 )
    {
      v27 = v26 - 1;
      v28 = *(_QWORD *)(a1 + 8);
      v15 = 1;
      v29 = v27 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v30 = 0;
      v19 = (unsigned int)(*(_DWORD *)(a1 + 16) + 1);
      v12 = v28 + 16LL * v29;
      v20 = *(_QWORD *)v12;
      if ( v8 != *(_QWORD *)v12 )
      {
        while ( v20 != -8 )
        {
          if ( !v30 && v20 == -16 )
            v30 = v12;
          a6 = v15 + 1;
          v29 = v27 & (v15 + v29);
          v12 = v28 + 16LL * v29;
          v20 = *(_QWORD *)v12;
          if ( v8 == *(_QWORD *)v12 )
            goto LABEL_15;
          v15 = a6;
        }
        if ( v30 )
          v12 = v30;
      }
      goto LABEL_15;
    }
LABEL_44:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v19;
  if ( *(_QWORD *)v12 != -8 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v12 = v8;
  *(_DWORD *)(v12 + 8) = 0;
  v21 = *a2;
  v32 = v34;
  v31 = v21;
  v33 = 0x200000000LL;
  sub_1BC35B0((__int64 *)(a1 + 32), (__int64)&v31, v19, v20, v15, a6);
  if ( v32 != v34 )
    _libc_free((unsigned __int64)v32);
  v16 = -858993459 * (unsigned int)((__int64)(*(_QWORD *)(a1 + 40) - *(_QWORD *)(a1 + 32)) >> 3) - 1;
  *(_DWORD *)(v12 + 8) = v16;
  return *(_QWORD *)(a1 + 32) + 40 * v16 + 8;
}
