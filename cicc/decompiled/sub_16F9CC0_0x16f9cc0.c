// Function: sub_16F9CC0
// Address: 0x16f9cc0
//
__int64 __fastcall sub_16F9CC0(__int64 a1, char a2)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // r12
  int v5; // r13d
  __int64 v6; // rcx
  __int64 v7; // r8
  char *v8; // rsi
  char *v9; // rax
  __int64 v10; // rdx
  __int64 *v11; // rax
  __m128i v12; // xmm0
  __int64 *v13; // r12
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // r12
  int v17; // r8d
  int v18; // r9d
  _QWORD *v19; // rdi
  unsigned __int64 v21; // rax
  __int64 v22; // r12
  __int64 v23; // rax
  const char *v24; // [rsp+0h] [rbp-70h] BYREF
  __m128i v25; // [rsp+8h] [rbp-68h] BYREF
  _QWORD *v26; // [rsp+18h] [rbp-58h]
  __int64 v27; // [rsp+20h] [rbp-50h]
  _QWORD v28[9]; // [rsp+28h] [rbp-48h] BYREF

  v3 = a1;
  v4 = *(_QWORD *)(a1 + 40);
  v5 = *(_DWORD *)(a1 + 60);
  sub_16F7930(a1, 1u);
  v8 = *(char **)(a1 + 40);
  while ( 1 )
  {
    v10 = (unsigned __int8)*v8;
    if ( (((*v8 & 0xDF) - 91) & 0xFD) == 0 )
      break;
    LOBYTE(v6) = (_BYTE)v10 == 58 || (_BYTE)v10 == 44;
    if ( (_BYTE)v6 )
      break;
    a1 = v3;
    v9 = sub_16F6460(v3, v8);
    v8 = v9;
    if ( *(char **)(v3 + 40) == v9 )
      break;
    ++*(_DWORD *)(v3 + 60);
    *(_QWORD *)(v3 + 40) = v9;
  }
  if ( (char *)v4 == v8 )
  {
    v24 = "Got empty alias or anchor";
    v21 = *(_QWORD *)(v3 + 48);
    v25.m128i_i16[4] = 259;
    if ( v21 <= v4 )
      *(_QWORD *)(v3 + 40) = v21 - 1;
    v22 = *(_QWORD *)(v3 + 344);
    if ( v22 )
    {
      v23 = sub_2241E50(a1, v8, v10, v6, v7);
      *(_DWORD *)v22 = 22;
      *(_QWORD *)(v22 + 8) = v23;
    }
    if ( !*(_BYTE *)(v3 + 74) )
      sub_16D14E0(*(__int64 **)v3, *(_QWORD *)(v3 + 40), 0, (__int64)&v24, 0, 0, 0, 0, *(_BYTE *)(v3 + 75));
    *(_BYTE *)(v3 + 74) = 1;
    return 0;
  }
  else
  {
    v25.m128i_i64[0] = v4;
    v26 = v28;
    v27 = 0;
    v25.m128i_i64[1] = (__int64)&v8[-v4];
    LOBYTE(v28[0]) = 0;
    LODWORD(v24) = (a2 == 0) + 20;
    v11 = (__int64 *)sub_145CBF0((__int64 *)(v3 + 80), 72, 16);
    v12 = _mm_loadu_si128(&v25);
    v13 = v11;
    *v11 = 0;
    v14 = v27;
    v11[1] = 0;
    LODWORD(v11) = (_DWORD)v24;
    *(__m128i *)(v13 + 3) = v12;
    *((_DWORD *)v13 + 4) = (_DWORD)v11;
    v13[5] = (__int64)(v13 + 7);
    sub_16F6740(v13 + 5, v28, (__int64)v28 + v14);
    v15 = *(_QWORD *)(v3 + 184);
    v13[1] = v3 + 184;
    v15 &= 0xFFFFFFFFFFFFFFF8LL;
    *v13 = v15 | *v13 & 7;
    *(_QWORD *)(v15 + 8) = v13;
    v16 = *(_QWORD *)(v3 + 184) & 7LL | (unsigned __int64)v13;
    *(_QWORD *)(v3 + 184) = v16;
    sub_16F79B0(v3, v16 & 0xFFFFFFFFFFFFFFF8LL, v5, 0, v17, v18);
    v19 = v26;
    *(_BYTE *)(v3 + 73) = 0;
    if ( v19 != v28 )
      j_j___libc_free_0(v19, v28[0] + 1LL);
    return 1;
  }
}
