// Function: sub_1F45B20
// Address: 0x1f45b20
//
__int64 __fastcall sub_1F45B20(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 result; // rax
  __m128i v8; // [rsp+0h] [rbp-100h] BYREF
  __int16 v9; // [rsp+10h] [rbp-F0h]
  __m128i v10; // [rsp+20h] [rbp-E0h] BYREF
  char v11; // [rsp+30h] [rbp-D0h]
  char v12; // [rsp+31h] [rbp-CFh]
  __m128i v13[2]; // [rsp+40h] [rbp-C0h] BYREF
  __m128i v14; // [rsp+60h] [rbp-A0h] BYREF
  __int16 v15; // [rsp+70h] [rbp-90h]
  __m128i v16[2]; // [rsp+80h] [rbp-80h] BYREF
  __m128i v17; // [rsp+A0h] [rbp-60h] BYREF
  char v18; // [rsp+B0h] [rbp-50h]
  char v19; // [rsp+B1h] [rbp-4Fh]
  __m128i v20[4]; // [rsp+C0h] [rbp-40h] BYREF

  v1 = sub_1F45960(qword_4FCBD20, qword_4FCBD28);
  if ( v1 )
    v1 = *(_QWORD *)(v1 + 32);
  *(_QWORD *)(a1 + 168) = v1;
  v2 = sub_1F45960(qword_4FCBE40, qword_4FCBE48);
  if ( v2 )
    v2 = *(_QWORD *)(v2 + 32);
  *(_QWORD *)(a1 + 176) = v2;
  v3 = sub_1F45960(qword_4FCBAE0, qword_4FCBAE8);
  if ( v3 )
    v3 = *(_QWORD *)(v3 + 32);
  *(_QWORD *)(a1 + 184) = v3;
  v4 = sub_1F45960(qword_4FCBC00, qword_4FCBC08);
  if ( !v4 )
  {
    v6 = *(_QWORD *)(a1 + 168);
    *(_QWORD *)(a1 + 192) = 0;
    if ( !v6 || !*(_QWORD *)(a1 + 176) )
      goto LABEL_12;
LABEL_15:
    v19 = 1;
    v17.m128i_i64[0] = (__int64)" specified!";
    v18 = 3;
    v15 = 257;
    if ( *off_4CD4AF8 )
    {
      v14.m128i_i64[0] = (__int64)off_4CD4AF8;
      LOBYTE(v15) = 3;
    }
    v12 = 1;
    v10.m128i_i64[0] = (__int64)" and ";
    v11 = 3;
    v9 = 257;
    if ( *off_4CD4AF0[0] )
    {
      v8.m128i_i64[0] = (__int64)off_4CD4AF0[0];
      LOBYTE(v9) = 3;
    }
    sub_14EC200(v13, &v8, &v10);
    sub_14EC200(v16, v13, &v14);
    sub_14EC200(v20, v16, &v17);
    sub_16BCFB0((__int64)v20, 1u);
  }
  v5 = *(_QWORD *)(v4 + 32);
  v6 = *(_QWORD *)(a1 + 168);
  *(_QWORD *)(a1 + 192) = v5;
  if ( v6 && *(_QWORD *)(a1 + 176) )
    goto LABEL_15;
  if ( *(_QWORD *)(a1 + 184) && v5 )
  {
    v19 = 1;
    v17.m128i_i64[0] = (__int64)" specified!";
    v18 = 3;
    v15 = 257;
    if ( *off_4CD4AE8[0] )
    {
      v14.m128i_i64[0] = (__int64)off_4CD4AE8[0];
      LOBYTE(v15) = 3;
    }
    v12 = 1;
    v10.m128i_i64[0] = (__int64)" and ";
    v11 = 3;
    v9 = 257;
    if ( *off_4CD4AE0[0] )
    {
      v8.m128i_i64[0] = (__int64)off_4CD4AE0[0];
      LOBYTE(v9) = 3;
    }
    sub_14EC200(v13, &v8, &v10);
    sub_14EC200(v16, v13, &v14);
    sub_14EC200(v20, v16, &v17);
    sub_16BCFB0((__int64)v20, 1u);
  }
LABEL_12:
  result = *(_QWORD *)(a1 + 176) | v6;
  *(_BYTE *)(a1 + 200) = result == 0;
  return result;
}
