// Function: sub_16FA830
// Address: 0x16fa830
//
__int64 __fastcall sub_16FA830(__int64 a1, _BYTE *a2, _DWORD *a3, _BYTE *a4)
{
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  unsigned int v12; // r12d
  __int64 *v14; // rax
  __m128i v15; // xmm0
  __int64 *v16; // r12
  __int64 v17; // rdx
  __int64 v18; // rdx
  _QWORD *v19; // rdi
  unsigned __int64 v20; // rax
  __int64 v21; // r13
  __int64 v22; // rax
  const char *v23; // [rsp+0h] [rbp-70h] BYREF
  __m128i v24; // [rsp+8h] [rbp-68h] BYREF
  _QWORD *v25; // [rsp+18h] [rbp-58h]
  __int64 v26; // [rsp+20h] [rbp-50h]
  _QWORD v27[9]; // [rsp+28h] [rbp-48h] BYREF

  v7 = *(_QWORD *)(a1 + 40);
  *a2 = sub_16F7CF0(a1);
  *a3 = sub_16F7D40(a1);
  if ( *a2 == 32 )
    *a2 = sub_16F7CF0(a1);
  *(_QWORD *)(a1 + 40) = sub_16F7770(a1, (char *)sub_16F6430, 0, *(_QWORD *)(a1 + 40));
  sub_16F7C20(a1);
  v8 = *(_QWORD *)(a1 + 40);
  if ( v8 == *(_QWORD *)(a1 + 48) )
  {
    LOBYTE(v27[0]) = 0;
    v24.m128i_i64[0] = v7;
    v25 = v27;
    v26 = 0;
    LODWORD(v23) = 19;
    v24.m128i_i64[1] = v8 - v7;
    v14 = (__int64 *)sub_145CBF0((__int64 *)(a1 + 80), 72, 16);
    v15 = _mm_loadu_si128(&v24);
    v16 = v14;
    *v14 = 0;
    v17 = v26;
    v14[1] = 0;
    LODWORD(v14) = (_DWORD)v23;
    *(__m128i *)(v16 + 3) = v15;
    *((_DWORD *)v16 + 4) = (_DWORD)v14;
    v16[5] = (__int64)(v16 + 7);
    sub_16F6740(v16 + 5, v27, (__int64)v27 + v17);
    v18 = *(_QWORD *)(a1 + 184);
    v16[1] = a1 + 184;
    v18 &= 0xFFFFFFFFFFFFFFF8LL;
    *v16 = v18 | *v16 & 7;
    *(_QWORD *)(v18 + 8) = v16;
    v19 = v25;
    *(_QWORD *)(a1 + 184) = (unsigned __int64)v16 | *(_QWORD *)(a1 + 184) & 7LL;
    *a4 = 1;
    if ( v19 != v27 )
      j_j___libc_free_0(v19, v27[0] + 1LL);
    return 1;
  }
  else
  {
    v12 = sub_16F7970(a1);
    if ( !(_BYTE)v12 )
    {
      v23 = "Expected a line break after block scalar header";
      v20 = *(_QWORD *)(a1 + 48);
      v24.m128i_i16[4] = 259;
      if ( *(_QWORD *)(a1 + 40) >= v20 )
        *(_QWORD *)(a1 + 40) = v20 - 1;
      v21 = *(_QWORD *)(a1 + 344);
      if ( v21 )
      {
        v22 = sub_2241E50(a1, sub_16F6430, v9, v10, v11);
        *(_DWORD *)v21 = 22;
        *(_QWORD *)(v21 + 8) = v22;
      }
      if ( !*(_BYTE *)(a1 + 74) )
        sub_16D14E0(*(__int64 **)a1, *(_QWORD *)(a1 + 40), 0, (__int64)&v23, 0, 0, 0, 0, *(_BYTE *)(a1 + 75));
      *(_BYTE *)(a1 + 74) = 1;
    }
  }
  return v12;
}
