// Function: sub_11AE3E0
// Address: 0x11ae3e0
//
__int64 __fastcall sub_11AE3E0(const __m128i *a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // eax
  unsigned __int64 v6; // rdx
  __m128i v7; // xmm1
  __int64 v8; // rax
  unsigned __int64 v9; // xmm2_8
  __m128i v10; // xmm3
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // rbx
  int v14; // r11d
  __int64 v15; // r9
  _QWORD *v16; // rdx
  unsigned int v17; // edi
  _QWORD *v18; // rax
  __int64 v19; // rcx
  unsigned int v20; // esi
  __int64 v21; // r13
  __int64 v22; // r8
  int v23; // esi
  int v24; // esi
  unsigned int v25; // ecx
  int v26; // eax
  __int64 v27; // rdi
  int v28; // r14d
  _QWORD *v29; // r10
  unsigned int v30; // r12d
  int v32; // eax
  __int64 v33; // rax
  __int64 v34; // r11
  int v35; // ecx
  int v36; // ecx
  __int64 v37; // rdi
  unsigned int v38; // r14d
  int v39; // r10d
  __int64 v40; // rsi
  unsigned int v41; // [rsp+0h] [rbp-B0h]
  unsigned int v42; // [rsp+0h] [rbp-B0h]
  __int64 v43; // [rsp+10h] [rbp-A0h]
  __int64 v44; // [rsp+18h] [rbp-98h]
  unsigned __int64 v45; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v46; // [rsp+28h] [rbp-88h]
  __m128i v47[2]; // [rsp+30h] [rbp-80h] BYREF
  unsigned __int64 v48; // [rsp+50h] [rbp-60h]
  __int64 v49; // [rsp+58h] [rbp-58h]
  __m128i v50; // [rsp+60h] [rbp-50h]
  __int64 v51; // [rsp+70h] [rbp-40h]

  v5 = *(_DWORD *)(a3 + 8);
  v46 = v5;
  if ( v5 > 0x40 )
  {
    sub_C43690((__int64)&v45, -1, 1);
  }
  else
  {
    v6 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v5;
    if ( !v5 )
      v6 = 0;
    v45 = v6;
  }
  v7 = _mm_loadu_si128(a1 + 7);
  v8 = a1[10].m128i_i64[0];
  v9 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v10 = _mm_loadu_si128(a1 + 9);
  v47[0] = _mm_loadu_si128(a1 + 6);
  v51 = v8;
  v48 = v9;
  v47[1] = v7;
  v49 = a2;
  v50 = v10;
  v11 = sub_11A7600(a1, (unsigned __int8 *)a2, (__int64)&v45, (unsigned __int64 *)a3, 0, v47);
  if ( !v11 )
  {
    v30 = 0;
    goto LABEL_21;
  }
  if ( a2 == v11 || (v12 = *(_QWORD *)(a2 + 16)) == 0 )
  {
    v30 = 1;
    goto LABEL_21;
  }
  v13 = a1[2].m128i_i64[1];
  v44 = v11;
  v43 = v13 + 2064;
  do
  {
    while ( 1 )
    {
      v20 = *(_DWORD *)(v13 + 2088);
      v21 = *(_QWORD *)(v12 + 24);
      v22 = *(unsigned int *)(v13 + 8);
      if ( !v20 )
      {
        ++*(_QWORD *)(v13 + 2064);
LABEL_13:
        v41 = v22;
        sub_9BAAD0(v43, 2 * v20);
        v23 = *(_DWORD *)(v13 + 2088);
        if ( !v23 )
          goto LABEL_64;
        v24 = v23 - 1;
        v15 = *(_QWORD *)(v13 + 2072);
        v22 = v41;
        v25 = v24 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v26 = *(_DWORD *)(v13 + 2080) + 1;
        v16 = (_QWORD *)(v15 + 16LL * v25);
        v27 = *v16;
        if ( v21 != *v16 )
        {
          v28 = 1;
          v29 = 0;
          while ( v27 != -4096 )
          {
            if ( v27 == -8192 && !v29 )
              v29 = v16;
            v25 = v24 & (v28 + v25);
            v16 = (_QWORD *)(v15 + 16LL * v25);
            v27 = *v16;
            if ( v21 == *v16 )
              goto LABEL_34;
            ++v28;
          }
          if ( v29 )
            v16 = v29;
        }
        goto LABEL_34;
      }
      v14 = 1;
      v15 = *(_QWORD *)(v13 + 2072);
      v16 = 0;
      v17 = (v20 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v18 = (_QWORD *)(v15 + 16LL * v17);
      v19 = *v18;
      if ( v21 != *v18 )
        break;
LABEL_10:
      v12 = *(_QWORD *)(v12 + 8);
      if ( !v12 )
        goto LABEL_39;
    }
    while ( v19 != -4096 )
    {
      if ( v19 != -8192 || v16 )
        v18 = v16;
      v17 = (v20 - 1) & (v14 + v17);
      v19 = *(_QWORD *)(v15 + 16LL * v17);
      if ( v21 == v19 )
        goto LABEL_10;
      ++v14;
      v16 = v18;
      v18 = (_QWORD *)(v15 + 16LL * v17);
    }
    if ( !v16 )
      v16 = v18;
    v32 = *(_DWORD *)(v13 + 2080);
    ++*(_QWORD *)(v13 + 2064);
    v26 = v32 + 1;
    if ( 4 * v26 >= 3 * v20 )
      goto LABEL_13;
    if ( v20 - *(_DWORD *)(v13 + 2084) - v26 <= v20 >> 3 )
    {
      v42 = v22;
      sub_9BAAD0(v43, v20);
      v35 = *(_DWORD *)(v13 + 2088);
      if ( !v35 )
      {
LABEL_64:
        ++*(_DWORD *)(v13 + 2080);
        BUG();
      }
      v36 = v35 - 1;
      v37 = *(_QWORD *)(v13 + 2072);
      v15 = 0;
      v38 = v36 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v22 = v42;
      v39 = 1;
      v26 = *(_DWORD *)(v13 + 2080) + 1;
      v16 = (_QWORD *)(v37 + 16LL * v38);
      v40 = *v16;
      if ( v21 != *v16 )
      {
        while ( v40 != -4096 )
        {
          if ( !v15 && v40 == -8192 )
            v15 = (__int64)v16;
          v38 = v36 & (v39 + v38);
          v16 = (_QWORD *)(v37 + 16LL * v38);
          v40 = *v16;
          if ( v21 == *v16 )
            goto LABEL_34;
          ++v39;
        }
        if ( v15 )
          v16 = (_QWORD *)v15;
      }
    }
LABEL_34:
    *(_DWORD *)(v13 + 2080) = v26;
    if ( *v16 != -4096 )
      --*(_DWORD *)(v13 + 2084);
    *v16 = v21;
    *((_DWORD *)v16 + 2) = v22;
    v33 = *(unsigned int *)(v13 + 8);
    if ( v33 + 1 > (unsigned __int64)*(unsigned int *)(v13 + 12) )
    {
      sub_C8D5F0(v13, (const void *)(v13 + 16), v33 + 1, 8u, v22, v15);
      v33 = *(unsigned int *)(v13 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v13 + 8 * v33) = v21;
    ++*(_DWORD *)(v13 + 8);
    v12 = *(_QWORD *)(v12 + 8);
  }
  while ( v12 );
LABEL_39:
  v34 = v44;
  if ( !*(_QWORD *)(v44 + 16)
    && *(_BYTE *)v44 > 0x1Cu
    && (*(_BYTE *)(v44 + 7) & 0x10) == 0
    && (*(_BYTE *)(a2 + 7) & 0x10) != 0 )
  {
    sub_BD6B90((unsigned __int8 *)v44, (unsigned __int8 *)a2);
    v34 = v44;
  }
  v30 = 1;
  sub_BD84D0(a2, v34);
LABEL_21:
  if ( v46 > 0x40 && v45 )
    j_j___libc_free_0_0(v45);
  return v30;
}
