// Function: sub_2E3D9B0
// Address: 0x2e3d9b0
//
__int64 __fastcall sub_2E3D9B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdi
  unsigned int v6; // r8d
  int v7; // r15d
  __int64 v8; // rcx
  __int64 *v9; // r10
  unsigned int v10; // r13d
  unsigned int v11; // eax
  unsigned int *v12; // rsi
  __int64 v13; // r14
  int v15; // eax
  int v16; // ecx
  unsigned __int64 v17; // rax
  const __m128i *v18; // rsi
  int v19; // eax
  int v20; // esi
  __int64 v21; // r8
  unsigned int v22; // eax
  __int64 v23; // rdi
  int v24; // r11d
  __int64 *v25; // r9
  int v26; // eax
  int v27; // eax
  __int64 *v28; // r8
  int v29; // r9d
  unsigned int v30; // r13d
  __int64 v31; // rdi
  __int64 v32; // rsi
  __int64 v33; // [rsp+8h] [rbp-48h]
  __int64 v34; // [rsp+8h] [rbp-48h]
  __int64 v35; // [rsp+8h] [rbp-48h]
  unsigned int v36[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v4 = a1 + 160;
  v6 = *(_DWORD *)(a1 + 184);
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 160);
    goto LABEL_23;
  }
  v7 = 1;
  v8 = *(_QWORD *)(a1 + 168);
  v9 = 0;
  v10 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v11 = (v6 - 1) & v10;
  v12 = (unsigned int *)(v8 + 16LL * v11);
  v13 = *(_QWORD *)v12;
  if ( *(_QWORD *)v12 == a2 )
    return sub_FE8AF0(a1, v12 + 2, a3);
  while ( v13 != -4096 )
  {
    if ( !v9 && v13 == -8192 )
      v9 = (__int64 *)v12;
    v11 = (v6 - 1) & (v7 + v11);
    v12 = (unsigned int *)(v8 + 16LL * v11);
    v13 = *(_QWORD *)v12;
    if ( *(_QWORD *)v12 == a2 )
      return sub_FE8AF0(a1, v12 + 2, a3);
    ++v7;
  }
  v15 = *(_DWORD *)(a1 + 176);
  if ( !v9 )
    v9 = (__int64 *)v12;
  ++*(_QWORD *)(a1 + 160);
  v16 = v15 + 1;
  if ( 4 * (v15 + 1) >= 3 * v6 )
  {
LABEL_23:
    v34 = a3;
    sub_2E3D7D0(v4, 2 * v6);
    v19 = *(_DWORD *)(a1 + 184);
    if ( v19 )
    {
      v20 = v19 - 1;
      v21 = *(_QWORD *)(a1 + 168);
      a3 = v34;
      v22 = (v19 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v16 = *(_DWORD *)(a1 + 176) + 1;
      v9 = (__int64 *)(v21 + 16LL * v22);
      v23 = *v9;
      if ( *v9 != a2 )
      {
        v24 = 1;
        v25 = 0;
        while ( v23 != -4096 )
        {
          if ( !v25 && v23 == -8192 )
            v25 = v9;
          v22 = v20 & (v24 + v22);
          v9 = (__int64 *)(v21 + 16LL * v22);
          v23 = *v9;
          if ( *v9 == a2 )
            goto LABEL_14;
          ++v24;
        }
        if ( v25 )
          v9 = v25;
      }
      goto LABEL_14;
    }
    goto LABEL_46;
  }
  if ( v6 - *(_DWORD *)(a1 + 180) - v16 <= v6 >> 3 )
  {
    v35 = a3;
    sub_2E3D7D0(v4, v6);
    v26 = *(_DWORD *)(a1 + 184);
    if ( v26 )
    {
      v27 = v26 - 1;
      v28 = 0;
      a3 = v35;
      v29 = 1;
      v30 = v27 & v10;
      v31 = *(_QWORD *)(a1 + 168);
      v16 = *(_DWORD *)(a1 + 176) + 1;
      v9 = (__int64 *)(v31 + 16LL * v30);
      v32 = *v9;
      if ( *v9 != a2 )
      {
        while ( v32 != -4096 )
        {
          if ( v32 == -8192 && !v28 )
            v28 = v9;
          v30 = v27 & (v29 + v30);
          v9 = (__int64 *)(v31 + 16LL * v30);
          v32 = *v9;
          if ( *v9 == a2 )
            goto LABEL_14;
          ++v29;
        }
        if ( v28 )
          v9 = v28;
      }
      goto LABEL_14;
    }
LABEL_46:
    ++*(_DWORD *)(a1 + 176);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(a1 + 176) = v16;
  if ( *v9 != -4096 )
    --*(_DWORD *)(a1 + 180);
  *v9 = a2;
  *((_DWORD *)v9 + 2) = -1;
  v17 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(a1 + 16) - *(_QWORD *)(a1 + 8)) >> 3);
  *((_DWORD *)v9 + 2) = v17;
  v18 = *(const __m128i **)(a1 + 16);
  v36[0] = v17;
  if ( v18 == *(const __m128i **)(a1 + 24) )
  {
    v33 = a3;
    sub_FDDD10((const __m128i **)(a1 + 8), v18);
    a3 = v33;
  }
  else
  {
    if ( v18 )
    {
      v18->m128i_i64[0] = 0;
      v18->m128i_i16[4] = 0;
      v18[1].m128i_i64[0] = 0;
      v18 = *(const __m128i **)(a1 + 16);
    }
    *(_QWORD *)(a1 + 16) = (char *)v18 + 24;
  }
  return sub_FE8AF0(a1, v36, a3);
}
