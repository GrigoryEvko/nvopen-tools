// Function: sub_16F9EC0
// Address: 0x16f9ec0
//
__int64 __fastcall sub_16F9EC0(__int64 a1, char a2)
{
  __int64 v2; // rbx
  __int64 v3; // r12
  int v4; // r13d
  unsigned __int64 v5; // rdx
  char *v6; // r8
  unsigned __int64 v7; // rcx
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 *v10; // rax
  __m128i v11; // xmm0
  __int64 *v12; // r12
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // r12
  int v16; // r8d
  int v17; // r9d
  _QWORD *v18; // rdi
  char v20; // al
  char *v21; // rax
  char *v22; // rax
  __int64 v23; // r12
  unsigned __int64 v24; // rdx
  __int64 v25; // rax
  const char *v26; // [rsp+0h] [rbp-60h] BYREF
  __m128i v27; // [rsp+8h] [rbp-58h] BYREF
  _QWORD *v28; // [rsp+18h] [rbp-48h]
  __int64 v29; // [rsp+20h] [rbp-40h]
  _QWORD v30[7]; // [rsp+28h] [rbp-38h] BYREF

  v2 = a1;
  v3 = *(_QWORD *)(a1 + 40);
  v4 = *(_DWORD *)(a1 + 60);
  if ( a2 )
  {
    v5 = *(_QWORD *)(a1 + 48);
    v6 = *(char **)(a1 + 40);
    v7 = v3 + 1;
    while ( 1 )
    {
      v8 = (__int64)(v6 + 1);
      *(_QWORD *)(v2 + 40) = v6 + 1;
      if ( v6 + 1 == (char *)v5 )
        break;
      while ( *(_BYTE *)v8 != 34 )
      {
        *(_QWORD *)(v2 + 40) = ++v8;
        if ( v8 == v5 )
          goto LABEL_25;
      }
      v6 = *(char **)(v2 + 40);
      if ( *(_BYTE *)(v8 - 1) == 92 )
      {
        a1 = v7;
        if ( sub_16F6490(v7, v8) )
          continue;
      }
      goto LABEL_9;
    }
LABEL_25:
    v23 = *(_QWORD *)(v2 + 344);
    v24 = v5 - 1;
    v26 = "Expected quote at end of scalar";
    v27.m128i_i16[4] = 259;
    *(_QWORD *)(v2 + 40) = v24;
    if ( v23 )
    {
      v25 = sub_2241E50(a1, v8, v24, v7, v6);
      *(_DWORD *)v23 = 22;
      *(_QWORD *)(v23 + 8) = v25;
    }
    if ( !*(_BYTE *)(v2 + 74) )
      sub_16D14E0(*(__int64 **)v2, *(_QWORD *)(v2 + 40), 0, (__int64)&v26, 0, 0, 0, 0, *(_BYTE *)(v2 + 75));
    *(_BYTE *)(v2 + 74) = 1;
    return 0;
  }
  v8 = 1;
  sub_16F7930(a1, 1u);
  v6 = *(char **)(a1 + 40);
  v5 = *(_QWORD *)(a1 + 48);
  while ( 1 )
  {
    while ( 1 )
    {
      v7 = (unsigned __int64)(v6 + 1);
      v20 = *v6;
      if ( v5 > (unsigned __int64)(v6 + 1) )
        goto LABEL_15;
LABEL_19:
      if ( v20 == 39 )
        goto LABEL_9;
LABEL_16:
      v8 = (__int64)v6;
      a1 = v2;
      v21 = sub_16F6380(v2, v6);
      v6 = v21;
      if ( *(char **)(v2 + 40) == v21 )
        break;
      v5 = *(_QWORD *)(v2 + 48);
      if ( (char *)v5 == v21 )
        goto LABEL_10;
      v7 = (unsigned __int64)(v21 + 1);
      ++*(_DWORD *)(v2 + 60);
      *(_QWORD *)(v2 + 40) = v21;
      v20 = *v21;
      if ( v5 <= (unsigned __int64)(v6 + 1) )
        goto LABEL_19;
LABEL_15:
      if ( v20 != 39 )
        goto LABEL_16;
      if ( v6[1] != 39 )
        goto LABEL_9;
      v8 = 2;
      a1 = v2;
      sub_16F7930(v2, 2u);
      v6 = *(char **)(v2 + 40);
      v5 = *(_QWORD *)(v2 + 48);
    }
    v8 = (__int64)v21;
    a1 = v2;
    v22 = sub_16F7720(v2, v21);
    v6 = v22;
    if ( *(char **)(v2 + 40) == v22 )
      break;
    ++*(_DWORD *)(v2 + 64);
    v5 = *(_QWORD *)(v2 + 48);
    *(_QWORD *)(v2 + 40) = v22;
    *(_DWORD *)(v2 + 60) = 0;
  }
  v5 = *(_QWORD *)(v2 + 48);
LABEL_9:
  if ( (char *)v5 == v6 )
    goto LABEL_25;
LABEL_10:
  sub_16F7930(v2, 1u);
  v9 = *(_QWORD *)(v2 + 40);
  v27.m128i_i64[0] = v3;
  v28 = v30;
  v29 = 0;
  LOBYTE(v30[0]) = 0;
  LODWORD(v26) = 18;
  v27.m128i_i64[1] = v9 - v3;
  v10 = (__int64 *)sub_145CBF0((__int64 *)(v2 + 80), 72, 16);
  v11 = _mm_loadu_si128(&v27);
  v12 = v10;
  *v10 = 0;
  v13 = v29;
  v10[1] = 0;
  LODWORD(v10) = (_DWORD)v26;
  *(__m128i *)(v12 + 3) = v11;
  *((_DWORD *)v12 + 4) = (_DWORD)v10;
  v12[5] = (__int64)(v12 + 7);
  sub_16F6740(v12 + 5, v30, (__int64)v30 + v13);
  v14 = *(_QWORD *)(v2 + 184);
  v12[1] = v2 + 184;
  v14 &= 0xFFFFFFFFFFFFFFF8LL;
  *v12 = v14 | *v12 & 7;
  *(_QWORD *)(v14 + 8) = v12;
  v15 = *(_QWORD *)(v2 + 184) & 7LL | (unsigned __int64)v12;
  *(_QWORD *)(v2 + 184) = v15;
  sub_16F79B0(v2, v15 & 0xFFFFFFFFFFFFFFF8LL, v4, 0, v16, v17);
  v18 = v28;
  *(_BYTE *)(v2 + 73) = 0;
  if ( v18 != v30 )
    j_j___libc_free_0(v18, v30[0] + 1LL);
  return 1;
}
