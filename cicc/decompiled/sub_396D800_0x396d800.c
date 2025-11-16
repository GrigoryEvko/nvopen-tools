// Function: sub_396D800
// Address: 0x396d800
//
__int64 __fastcall sub_396D800(__int64 a1)
{
  __int64 v1; // rsi
  char v2; // al
  __int64 v4; // rdx
  __int64 v5; // rdx
  __int64 v6; // [rsp+0h] [rbp-C0h] BYREF
  __int64 v7; // [rsp+8h] [rbp-B8h]
  __m128i v8; // [rsp+10h] [rbp-B0h] BYREF
  __int16 v9; // [rsp+20h] [rbp-A0h]
  __m128i v10; // [rsp+30h] [rbp-90h] BYREF
  char v11; // [rsp+40h] [rbp-80h]
  char v12; // [rsp+41h] [rbp-7Fh]
  __m128i v13[2]; // [rsp+50h] [rbp-70h] BYREF
  __m128i v14; // [rsp+70h] [rbp-50h] BYREF
  char v15; // [rsp+80h] [rbp-40h]
  char v16; // [rsp+81h] [rbp-3Fh]
  __m128i v17[3]; // [rsp+90h] [rbp-30h] BYREF

  v1 = *(_QWORD *)(a1 + 304);
  if ( (*(_BYTE *)(v1 + 8) & 2) != 0 )
  {
    v2 = *(_BYTE *)(v1 + 9);
    if ( (v2 & 0xC) == 8 )
    {
      *(_QWORD *)(v1 + 24) = 0;
      *(_BYTE *)(v1 + 9) = v2 & 0xF3;
    }
    *(_QWORD *)v1 &= 7uLL;
    *(_BYTE *)(v1 + 8) &= ~2u;
    v1 = *(_QWORD *)(a1 + 304);
  }
  if ( (*(_BYTE *)(v1 + 9) & 0xC) == 8 )
  {
    v16 = 1;
    v15 = 3;
    v14.m128i_i64[0] = (__int64)"' is a protected alias";
    v12 = 1;
    v6 = sub_3913870((_BYTE *)v1);
    v7 = v4;
    v8.m128i_i64[0] = (__int64)&v6;
    v10.m128i_i64[0] = (__int64)"'";
    v9 = 261;
    v11 = 3;
    sub_14EC200(v13, &v10, &v8);
    sub_14EC200(v17, v13, &v14);
    sub_16BCFB0((__int64)v17, 1u);
  }
  if ( (*(_QWORD *)v1 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    v16 = 1;
    v15 = 3;
    v14.m128i_i64[0] = (__int64)"' label emitted multiple times to assembly file";
    v12 = 1;
    v6 = sub_3913870((_BYTE *)v1);
    v7 = v5;
    v8.m128i_i64[0] = (__int64)&v6;
    v10.m128i_i64[0] = (__int64)"'";
    v9 = 261;
    v11 = 3;
    sub_14EC200(v13, &v10, &v8);
    sub_14EC200(v17, v13, &v14);
    sub_16BCFB0((__int64)v17, 1u);
  }
  return (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 256) + 176LL))(
           *(_QWORD *)(a1 + 256),
           v1,
           0);
}
