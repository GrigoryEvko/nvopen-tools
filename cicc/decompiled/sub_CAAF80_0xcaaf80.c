// Function: sub_CAAF80
// Address: 0xcaaf80
//
__int64 __fastcall sub_CAAF80(__int64 a1, char a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // r13
  int v7; // r14d
  char *v8; // r12
  char *v9; // rdx
  unsigned __int64 v10; // rcx
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned __int64 v14; // rsi
  unsigned __int64 v15; // r12
  __m128i v16; // xmm0
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rsi
  __int64 v20; // r8
  __int64 v21; // r9
  _QWORD *v22; // rdi
  unsigned __int64 v24; // rax
  char v25; // dl
  char *v26; // rax
  char *v27; // rax
  __int64 v28; // r13
  __int64 v29; // rax
  const char *v30; // [rsp+0h] [rbp-70h] BYREF
  __m128i v31; // [rsp+8h] [rbp-68h] BYREF
  _BYTE *v32; // [rsp+18h] [rbp-58h]
  __int64 v33; // [rsp+20h] [rbp-50h]
  _QWORD v34[9]; // [rsp+28h] [rbp-48h] BYREF

  v5 = a1;
  v6 = *(_QWORD *)(a1 + 40);
  v7 = *(_DWORD *)(a1 + 60);
  if ( a2 )
  {
    v8 = *(char **)(a1 + 48);
    v9 = *(char **)(a1 + 40);
    v10 = v6 + 1;
    while ( 1 )
    {
      v11 = (__int64)(v9 + 1);
      *(_QWORD *)(v5 + 40) = v9 + 1;
      if ( v9 + 1 == v8 )
        goto LABEL_29;
      while ( *(_BYTE *)v11 != 34 )
      {
        *(_QWORD *)(v5 + 40) = ++v11;
        if ( (char *)v11 == v8 )
          goto LABEL_29;
      }
      v9 = *(char **)(v5 + 40);
      if ( *(_BYTE *)(v11 - 1) == 92 )
      {
        a1 = v10;
        if ( sub_CA6160(v10, v11) )
          continue;
      }
      goto LABEL_9;
    }
  }
  v11 = 1;
  sub_CA7F70(a1, 1u);
  v8 = *(char **)(a1 + 40);
  v24 = *(_QWORD *)(a1 + 48);
  if ( v8 == (char *)v24 )
  {
LABEL_29:
    v28 = *(_QWORD *)(v5 + 336);
    v30 = "Expected quote at end of scalar";
    LOWORD(v33) = 259;
    if ( v28 )
    {
      v29 = sub_2241E50(a1, v11, v9, v10, a5);
      *(_DWORD *)v28 = 22;
      *(_QWORD *)(v28 + 8) = v29;
    }
    if ( !*(_BYTE *)(v5 + 75) )
      sub_C91CB0(*(__int64 **)v5, (unsigned __int64)(v8 - 1), 0, (__int64)&v30, 0, 0, 0, 0, *(_BYTE *)(v5 + 76));
    *(_BYTE *)(v5 + 75) = 1;
    return 0;
  }
  while ( 1 )
  {
    v10 = (unsigned __int64)(v8 + 1);
    v25 = *v8;
    if ( v24 > (unsigned __int64)(v8 + 1) )
      break;
    while ( 2 )
    {
      if ( v25 == 39 )
        goto LABEL_25;
LABEL_19:
      v11 = (__int64)v8;
      a1 = v5;
      v26 = sub_CA6050(v5, v8);
      v8 = v26;
      if ( *(char **)(v5 + 40) != v26 )
      {
        v24 = *(_QWORD *)(v5 + 48);
        if ( (char *)v24 == v8 )
          goto LABEL_10;
        v10 = (unsigned __int64)(v8 + 1);
        ++*(_DWORD *)(v5 + 60);
        *(_QWORD *)(v5 + 40) = v8;
        v25 = *v8;
        if ( v24 <= (unsigned __int64)(v8 + 1) )
          continue;
        goto LABEL_18;
      }
      break;
    }
    v11 = (__int64)v26;
    a1 = v5;
    v27 = sub_CA7C80(v5, v26);
    v8 = v27;
    if ( *(char **)(v5 + 40) == v27 )
    {
      v8 = *(char **)(v5 + 48);
      v9 = v27;
      goto LABEL_9;
    }
    ++*(_DWORD *)(v5 + 64);
    *(_QWORD *)(v5 + 40) = v27;
    *(_DWORD *)(v5 + 60) = 0;
LABEL_28:
    v24 = *(_QWORD *)(v5 + 48);
    if ( (char *)v24 == v8 )
      goto LABEL_29;
  }
LABEL_18:
  if ( v25 != 39 )
    goto LABEL_19;
  if ( v8[1] == 39 )
  {
    v11 = 2;
    a1 = v5;
    sub_CA7F70(v5, 2u);
    v8 = *(char **)(v5 + 40);
    goto LABEL_28;
  }
LABEL_25:
  v9 = *(char **)(v5 + 40);
  v8 = (char *)v24;
LABEL_9:
  if ( v9 == v8 )
    goto LABEL_29;
LABEL_10:
  sub_CA7F70(v5, 1u);
  v12 = *(_QWORD *)(v5 + 40);
  v31.m128i_i64[0] = v6;
  *(_QWORD *)(v5 + 160) += 72LL;
  v32 = v34;
  v31.m128i_i64[1] = v12 - v6;
  v13 = *(_QWORD *)(v5 + 80);
  v33 = 0;
  LOBYTE(v34[0]) = 0;
  v14 = (v13 + 15) & 0xFFFFFFFFFFFFFFF0LL;
  LODWORD(v30) = 18;
  if ( *(_QWORD *)(v5 + 88) >= v14 + 72 && v13 )
  {
    *(_QWORD *)(v5 + 80) = v14 + 72;
    v15 = (v13 + 15) & 0xFFFFFFFFFFFFFFF0LL;
    if ( !v14 )
    {
      MEMORY[8] = v5 + 176;
      BUG();
    }
  }
  else
  {
    v15 = sub_9D1E70(v5 + 80, 72, 72, 4);
  }
  *(_QWORD *)v15 = 0;
  *(_QWORD *)(v15 + 8) = 0;
  *(_DWORD *)(v15 + 16) = (_DWORD)v30;
  v16 = _mm_loadu_si128(&v31);
  *(_QWORD *)(v15 + 40) = v15 + 56;
  *(__m128i *)(v15 + 24) = v16;
  sub_CA64F0((__int64 *)(v15 + 40), v32, (__int64)&v32[v33]);
  v17 = *(_QWORD *)v15;
  v18 = *(_QWORD *)(v5 + 176);
  *(_QWORD *)(v15 + 8) = v5 + 176;
  v18 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v15 = v18 | v17 & 7;
  *(_QWORD *)(v18 + 8) = v15;
  v19 = v15 | *(_QWORD *)(v5 + 176) & 7LL;
  *(_QWORD *)(v5 + 176) = v19;
  sub_CA80E0(v5, v19 & 0xFFFFFFFFFFFFFFF8LL, v7, 0, v20, v21);
  v22 = v32;
  *(_WORD *)(v5 + 73) = 256;
  if ( v22 != v34 )
    j_j___libc_free_0(v22, v34[0] + 1LL);
  return 1;
}
