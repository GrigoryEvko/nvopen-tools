// Function: sub_CAAD30
// Address: 0xcaad30
//
__int64 __fastcall sub_CAAD30(__int64 a1, char a2)
{
  __int64 v2; // r14
  unsigned __int64 v4; // r12
  int v5; // r13d
  __int64 v6; // rcx
  __int64 v7; // r8
  char *v8; // rsi
  char *v9; // rax
  __int64 v10; // rax
  unsigned __int64 v11; // rbx
  __m128i v12; // xmm0
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rbx
  __int64 v16; // r8
  __int64 v17; // r9
  _QWORD *v18; // rdi
  __int64 v20; // rbx
  unsigned __int64 v21; // rax
  __int64 v22; // rax
  const char *v23; // [rsp+0h] [rbp-70h] BYREF
  __m128i v24; // [rsp+8h] [rbp-68h] BYREF
  _BYTE *v25; // [rsp+18h] [rbp-58h]
  __int64 v26; // [rsp+20h] [rbp-50h]
  _QWORD v27[9]; // [rsp+28h] [rbp-48h] BYREF

  v2 = a1;
  v4 = *(_QWORD *)(a1 + 40);
  v5 = *(_DWORD *)(a1 + 60);
  sub_CA7F70(a1, 1u);
  v8 = *(char **)(a1 + 40);
  if ( v8 != *(char **)(a1 + 48) )
  {
    while ( (((*v8 & 0xDF) - 91) & 0xFD) != 0 )
    {
      LOBYTE(v6) = *v8 == 58 || *v8 == 44;
      if ( (_BYTE)v6 )
        break;
      a1 = v2;
      v9 = sub_CA6130(v2, v8);
      v8 = v9;
      if ( *(char **)(v2 + 40) != v9 )
      {
        ++*(_DWORD *)(v2 + 60);
        *(_QWORD *)(v2 + 40) = v9;
        if ( *(char **)(v2 + 48) != v9 )
          continue;
      }
      goto LABEL_8;
    }
    v8 = *(char **)(v2 + 40);
  }
LABEL_8:
  if ( v8 == (char *)(v4 + 1) )
  {
    v20 = *(_QWORD *)(v2 + 336);
    v23 = "Got empty alias or anchor";
    v21 = *(_QWORD *)(v2 + 48);
    LOWORD(v26) = 259;
    if ( v4 >= v21 )
      v4 = v21 - 1;
    if ( v20 )
    {
      v22 = sub_2241E50(a1, v8, v21 - 1, v6, v7);
      *(_DWORD *)v20 = 22;
      *(_QWORD *)(v20 + 8) = v22;
    }
    if ( !*(_BYTE *)(v2 + 75) )
      sub_C91CB0(*(__int64 **)v2, v4, 0, (__int64)&v23, 0, 0, 0, 0, *(_BYTE *)(v2 + 76));
    *(_BYTE *)(v2 + 75) = 1;
    return 0;
  }
  else
  {
    v24.m128i_i64[0] = v4;
    *(_QWORD *)(v2 + 160) += 72LL;
    LODWORD(v23) = (a2 == 0) + 20;
    v10 = *(_QWORD *)(v2 + 80);
    v25 = v27;
    v26 = 0;
    v11 = (v10 + 15) & 0xFFFFFFFFFFFFFFF0LL;
    LOBYTE(v27[0]) = 0;
    v24.m128i_i64[1] = (__int64)&v8[-v4];
    if ( *(_QWORD *)(v2 + 88) >= v11 + 72 && v10 )
    {
      *(_QWORD *)(v2 + 80) = v11 + 72;
      if ( !v11 )
      {
        MEMORY[8] = v2 + 176;
        BUG();
      }
    }
    else
    {
      v11 = sub_9D1E70(v2 + 80, 72, 72, 4);
    }
    *(_QWORD *)v11 = 0;
    *(_QWORD *)(v11 + 8) = 0;
    *(_DWORD *)(v11 + 16) = (_DWORD)v23;
    v12 = _mm_loadu_si128(&v24);
    *(_QWORD *)(v11 + 40) = v11 + 56;
    *(__m128i *)(v11 + 24) = v12;
    sub_CA64F0((__int64 *)(v11 + 40), v25, (__int64)&v25[v26]);
    v13 = *(_QWORD *)v11;
    v14 = *(_QWORD *)(v2 + 176);
    *(_QWORD *)(v11 + 8) = v2 + 176;
    v14 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v11 = v14 | v13 & 7;
    *(_QWORD *)(v14 + 8) = v11;
    v15 = *(_QWORD *)(v2 + 176) & 7LL | v11;
    *(_QWORD *)(v2 + 176) = v15;
    sub_CA80E0(v2, v15 & 0xFFFFFFFFFFFFFFF8LL, v5, 0, v16, v17);
    v18 = v25;
    *(_WORD *)(v2 + 73) = 0;
    if ( v18 != v27 )
      j_j___libc_free_0(v18, v27[0] + 1LL);
    return 1;
  }
}
