// Function: sub_18D3440
// Address: 0x18d3440
//
__int64 __fastcall sub_18D3440(__int64 a1, __int64 *a2)
{
  __int64 v4; // r12
  unsigned int v5; // esi
  __int64 v6; // rdi
  int v7; // r11d
  _QWORD *v8; // rdx
  unsigned int v9; // ecx
  _QWORD *v10; // rax
  __int64 v11; // r9
  int v13; // eax
  int v14; // ecx
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // r13
  int v18; // eax
  int v19; // esi
  __int64 v20; // r8
  unsigned int v21; // eax
  __int64 v22; // rdi
  int v23; // r10d
  _QWORD *v24; // r9
  int v25; // eax
  int v26; // eax
  __int64 v27; // rdi
  int v28; // r9d
  unsigned int v29; // r14d
  _QWORD *v30; // r8
  __int64 v31; // rsi
  _QWORD v32[18]; // [rsp+10h] [rbp-150h] BYREF
  __int64 v33; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v34; // [rsp+A8h] [rbp-B8h]
  __int64 v35; // [rsp+B0h] [rbp-B0h]
  _QWORD v36[2]; // [rsp+B8h] [rbp-A8h] BYREF
  unsigned __int64 v37; // [rsp+C8h] [rbp-98h]
  _BYTE v38[16]; // [rsp+E0h] [rbp-80h] BYREF
  _QWORD v39[2]; // [rsp+F0h] [rbp-70h] BYREF
  unsigned __int64 v40; // [rsp+100h] [rbp-60h]
  _BYTE v41[16]; // [rsp+118h] [rbp-48h] BYREF
  char v42; // [rsp+128h] [rbp-38h]

  v4 = *a2;
  v5 = *(_DWORD *)(a1 + 24);
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_31;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v10 = (_QWORD *)(v6 + 16LL * v9);
  v11 = *v10;
  if ( v4 == *v10 )
    return *(_QWORD *)(a1 + 32) + 144LL * v10[1] + 8;
  while ( v11 != -8 )
  {
    if ( v11 == -16 && !v8 )
      v8 = v10;
    v9 = (v5 - 1) & (v7 + v9);
    v10 = (_QWORD *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( v4 == *v10 )
      return *(_QWORD *)(a1 + 32) + 144LL * v10[1] + 8;
    ++v7;
  }
  if ( !v8 )
    v8 = v10;
  v13 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v14 = v13 + 1;
  if ( 4 * (v13 + 1) >= 3 * v5 )
  {
LABEL_31:
    sub_14C5870(a1, 2 * v5);
    v18 = *(_DWORD *)(a1 + 24);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(a1 + 8);
      v21 = (v18 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v14 = *(_DWORD *)(a1 + 16) + 1;
      v8 = (_QWORD *)(v20 + 16LL * v21);
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
          v8 = (_QWORD *)(v20 + 16LL * v21);
          v22 = *v8;
          if ( v4 == *v8 )
            goto LABEL_14;
          ++v23;
        }
        if ( v24 )
          v8 = v24;
      }
      goto LABEL_14;
    }
    goto LABEL_54;
  }
  if ( v5 - *(_DWORD *)(a1 + 20) - v14 <= v5 >> 3 )
  {
    sub_14C5870(a1, v5);
    v25 = *(_DWORD *)(a1 + 24);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(a1 + 8);
      v28 = 1;
      v29 = v26 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v30 = 0;
      v14 = *(_DWORD *)(a1 + 16) + 1;
      v8 = (_QWORD *)(v27 + 16LL * v29);
      v31 = *v8;
      if ( v4 != *v8 )
      {
        while ( v31 != -8 )
        {
          if ( !v30 && v31 == -16 )
            v30 = v8;
          v29 = v26 & (v28 + v29);
          v8 = (_QWORD *)(v27 + 16LL * v29);
          v31 = *v8;
          if ( v4 == *v8 )
            goto LABEL_14;
          ++v28;
        }
        if ( v30 )
          v8 = v30;
      }
      goto LABEL_14;
    }
LABEL_54:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(a1 + 16) = v14;
  if ( *v8 != -8 )
    --*(_DWORD *)(a1 + 20);
  *v8 = v4;
  v8[1] = 0;
  v15 = *(_QWORD *)(a1 + 40) - *(_QWORD *)(a1 + 32);
  v8[1] = 0x8E38E38E38E38E39LL * (v15 >> 4);
  memset(v32, 0, 0x88u);
  v32[3] = &v32[7];
  v32[4] = &v32[7];
  v32[10] = &v32[14];
  v32[11] = &v32[14];
  v16 = *a2;
  v32[5] = 2;
  v33 = v16;
  v32[12] = 2;
  v34 = 0;
  v35 = 0;
  sub_16CCEE0(v36, (__int64)v38, 2, (__int64)&v32[2]);
  sub_16CCEE0(v39, (__int64)v41, 2, (__int64)&v32[9]);
  v17 = *(_QWORD *)(a1 + 40);
  v42 = v32[16];
  if ( v17 == *(_QWORD *)(a1 + 48) )
  {
    sub_18D1E40((__int64 *)(a1 + 32), (char *)v17, (__int64)&v33);
  }
  else
  {
    if ( v17 )
    {
      *(_QWORD *)v17 = v33;
      *(_WORD *)(v17 + 8) = v34;
      *(_QWORD *)(v17 + 16) = v35;
      sub_16CCEE0((_QWORD *)(v17 + 24), v17 + 64, 2, (__int64)v36);
      sub_16CCEE0((_QWORD *)(v17 + 80), v17 + 120, 2, (__int64)v39);
      *(_BYTE *)(v17 + 136) = v42;
      v17 = *(_QWORD *)(a1 + 40);
    }
    *(_QWORD *)(a1 + 40) = v17 + 144;
  }
  if ( v40 != v39[1] )
    _libc_free(v40);
  if ( v37 != v36[1] )
    _libc_free(v37);
  if ( v32[11] != v32[10] )
    _libc_free(v32[11]);
  if ( v32[4] != v32[3] )
    _libc_free(v32[4]);
  return *(_QWORD *)(a1 + 32) + v15 + 8;
}
