// Function: sub_CABAC0
// Address: 0xcabac0
//
__int64 __fastcall sub_CABAC0(__int64 a1, _BYTE *a2, _DWORD *a3, _BYTE *a4)
{
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  unsigned int v11; // r12d
  __int64 v13; // rdx
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // r12
  __m128i v16; // xmm0
  __int64 v17; // rdx
  __int64 v18; // rax
  _QWORD *v19; // rdi
  unsigned __int64 v20; // r13
  __int64 v21; // r14
  unsigned __int64 v22; // rax
  __int64 v23; // rax
  const char *v24; // [rsp+0h] [rbp-70h] BYREF
  __m128i v25; // [rsp+8h] [rbp-68h] BYREF
  _BYTE *v26; // [rsp+18h] [rbp-58h]
  __int64 v27; // [rsp+20h] [rbp-50h]
  _QWORD v28[9]; // [rsp+28h] [rbp-48h] BYREF

  v7 = *(_QWORD *)(a1 + 40);
  *a2 = sub_CA84E0(a1);
  *a3 = sub_CA8530(a1);
  if ( *a2 == 32 )
    *a2 = sub_CA84E0(a1);
  *(_QWORD *)(a1 + 40) = sub_CA7CD0(a1, (char *)sub_CA6100, 0, *(_QWORD *)(a1 + 40));
  sub_CA83A0(a1);
  v8 = *(_QWORD *)(a1 + 40);
  if ( v8 == *(_QWORD *)(a1 + 48) )
  {
    v13 = *(_QWORD *)(a1 + 80);
    v25.m128i_i64[0] = v7;
    v25.m128i_i64[1] = v8 - v7;
    *(_QWORD *)(a1 + 160) += 72LL;
    v14 = (v13 + 15) & 0xFFFFFFFFFFFFFFF0LL;
    v26 = v28;
    v27 = 0;
    LOBYTE(v28[0]) = 0;
    LODWORD(v24) = 19;
    if ( *(_QWORD *)(a1 + 88) >= v14 + 72 && v13 )
    {
      *(_QWORD *)(a1 + 80) = v14 + 72;
      v15 = (v13 + 15) & 0xFFFFFFFFFFFFFFF0LL;
      if ( !v14 )
      {
        MEMORY[8] = a1 + 176;
        BUG();
      }
    }
    else
    {
      v15 = sub_9D1E70(a1 + 80, 72, 72, 4);
    }
    *(_QWORD *)v15 = 0;
    *(_QWORD *)(v15 + 8) = 0;
    *(_DWORD *)(v15 + 16) = (_DWORD)v24;
    v16 = _mm_loadu_si128(&v25);
    *(_QWORD *)(v15 + 40) = v15 + 56;
    *(__m128i *)(v15 + 24) = v16;
    sub_CA64F0((__int64 *)(v15 + 40), v26, (__int64)&v26[v27]);
    v17 = *(_QWORD *)(a1 + 176);
    v18 = *(_QWORD *)v15;
    *(_QWORD *)(v15 + 8) = a1 + 176;
    v17 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v15 = v17 | v18 & 7;
    *(_QWORD *)(v17 + 8) = v15;
    v19 = v26;
    *(_QWORD *)(a1 + 176) = v15 | *(_QWORD *)(a1 + 176) & 7LL;
    *a4 = 1;
    if ( v19 != v28 )
      j_j___libc_free_0(v19, v28[0] + 1LL);
    return 1;
  }
  else
  {
    v11 = sub_CA80A0(a1);
    if ( !(_BYTE)v11 )
    {
      v20 = *(_QWORD *)(a1 + 40);
      v21 = *(_QWORD *)(a1 + 336);
      v24 = "Expected a line break after block scalar header";
      v22 = *(_QWORD *)(a1 + 48);
      LOWORD(v27) = 259;
      if ( v20 >= v22 )
        v20 = v22 - 1;
      if ( v21 )
      {
        v23 = sub_2241E50(a1, sub_CA6100, v22 - 1, v9, v10);
        *(_DWORD *)v21 = 22;
        *(_QWORD *)(v21 + 8) = v23;
      }
      if ( !*(_BYTE *)(a1 + 75) )
        sub_C91CB0(*(__int64 **)a1, v20, 0, (__int64)&v24, 0, 0, 0, 0, *(_BYTE *)(a1 + 76));
      *(_BYTE *)(a1 + 75) = 1;
    }
  }
  return v11;
}
