// Function: sub_38DBF80
// Address: 0x38dbf80
//
__int64 __fastcall sub_38DBF80(__int64 a1, _DWORD *a2, __int64 a3, __int64 a4)
{
  unsigned int v5; // r8d
  _BYTE *v7; // rdx
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // rdx
  int v16; // ecx
  __m128i v17; // xmm0
  _QWORD v18[2]; // [rsp+0h] [rbp-C0h] BYREF
  __m128i v19; // [rsp+10h] [rbp-B0h] BYREF
  _QWORD v20[2]; // [rsp+20h] [rbp-A0h] BYREF
  __int16 v21; // [rsp+30h] [rbp-90h]
  _QWORD v22[2]; // [rsp+40h] [rbp-80h] BYREF
  __int16 v23; // [rsp+50h] [rbp-70h]
  _BYTE *v24[2]; // [rsp+60h] [rbp-60h] BYREF
  __int64 v25; // [rsp+70h] [rbp-50h] BYREF
  __m128i v26; // [rsp+80h] [rbp-40h]
  __int64 v27; // [rsp+90h] [rbp-30h] BYREF
  __int64 v28; // [rsp+98h] [rbp-28h]

  if ( a4 != *(_QWORD *)(*(_QWORD *)(a1 + 32) + 24LL) )
  {
    v5 = *(_DWORD *)(a4 + 172);
    if ( v5 == -1 )
    {
      v5 = (*a2)++;
      *(_DWORD *)(a4 + 172) = v5;
    }
    if ( (*(_BYTE *)(a4 + 169) & 0x10) == 0 )
    {
      v7 = 0;
      return sub_38C23D0(a1, a3, v7, v5);
    }
    v7 = *(_BYTE **)(a4 + 176);
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 16) + 21LL) )
      return sub_38C23D0(a1, a3, v7, v5);
    v9 = *(_QWORD *)(a4 + 152);
    LOBYTE(v24[0]) = 36;
    v10 = *(_QWORD *)(a4 + 160);
    v19.m128i_i64[0] = v9;
    v19.m128i_i64[1] = v10;
    v11 = sub_16D20C0(v19.m128i_i64, (char *)v24, 1u, 0);
    if ( v11 == -1 )
    {
      v17 = _mm_loadu_si128(&v19);
      v27 = 0;
      v28 = 0;
      v26 = v17;
    }
    else
    {
      v12 = v11 + 1;
      if ( v11 + 1 > v19.m128i_i64[1] )
        v12 = v19.m128i_i64[1];
      v13 = v19.m128i_i64[1] - v12;
      v14 = v19.m128i_i64[0] + v12;
      if ( v11 && v11 > v19.m128i_i64[1] )
        v11 = v19.m128i_u64[1];
      v26.m128i_i64[0] = v19.m128i_i64[0];
      v26.m128i_i64[1] = v11;
      v27 = v14;
      v28 = v13;
    }
    v15 = *(_QWORD *)(a3 + 152);
    v18[1] = *(_QWORD *)(a3 + 160);
    v21 = 773;
    v20[0] = v18;
    v20[1] = "$";
    v18[0] = v15;
    v22[0] = v20;
    v22[1] = &v27;
    v23 = 1282;
    sub_16E2FC0((__int64 *)v24, (__int64)v22);
    v16 = *(_DWORD *)(a3 + 168);
    BYTE1(v16) |= 0x10u;
    a3 = sub_38C20E0(
           a1,
           v24[0],
           (__int64)v24[1],
           v16,
           *(_BYTE *)(a3 + 148),
           2u,
           (__int64)byte_3F871B3,
           0,
           0xFFFFFFFF,
           0);
    if ( (__int64 *)v24[0] != &v25 )
      j_j___libc_free_0((unsigned __int64)v24[0]);
  }
  return a3;
}
