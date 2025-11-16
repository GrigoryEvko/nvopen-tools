// Function: sub_2048A00
// Address: 0x2048a00
//
__int64 __fastcall sub_2048A00(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, _QWORD *a5)
{
  __int64 v9; // r12
  __int64 v10; // rdi
  char v11; // al
  __int64 v12; // rax
  unsigned int v13; // edx
  unsigned __int8 v14; // al
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 result; // rax
  __int64 v18; // r12
  unsigned int v19; // eax
  __int32 v20; // eax
  unsigned int v21; // esi
  unsigned int v22; // edx
  unsigned __int8 v23; // al
  __m128i v24; // rax
  __m128i si128; // xmm0
  __int64 v26; // rdx
  bool v27; // [rsp+8h] [rbp-C8h]
  __int64 v28; // [rsp+8h] [rbp-C8h]
  unsigned int v30; // [rsp+10h] [rbp-C0h]
  __int64 v31; // [rsp+18h] [rbp-B8h]
  __int64 v32; // [rsp+18h] [rbp-B8h]
  __m128i v33[2]; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v34; // [rsp+40h] [rbp-90h]
  __int64 v35; // [rsp+48h] [rbp-88h]
  __int64 v36; // [rsp+50h] [rbp-80h]
  __int64 v37; // [rsp+58h] [rbp-78h]
  _QWORD v38[4]; // [rsp+60h] [rbp-70h] BYREF
  __int128 v39; // [rsp+80h] [rbp-50h] BYREF
  __int64 v40; // [rsp+90h] [rbp-40h]

  v9 = a4[29];
  v10 = a5[4];
  v11 = *(_BYTE *)(v9 + 16);
  if ( (unsigned __int8)(v11 - 12) <= 2u || v11 == 8 )
  {
    v12 = sub_1E0A0C0(v10);
    v13 = 8 * sub_15A9520(v12, 0);
    if ( v13 == 32 )
    {
      v14 = 5;
    }
    else if ( v13 > 0x20 )
    {
      v14 = 6;
      if ( v13 != 64 )
      {
        v14 = 0;
        if ( v13 == 128 )
          v14 = 7;
      }
    }
    else
    {
      v14 = 3;
      if ( v13 != 8 )
        v14 = 4 * (v13 == 16);
    }
    v15 = sub_1D2A150((__int64)a5, (__int64 *)v9, v14, 0, 0, 0, 0, 0);
    v37 = v16;
    v36 = v15;
    a4[31] = v15;
    *((_DWORD *)a4 + 64) = v37;
    return a1;
  }
  else
  {
    v27 = v11 == 8;
    v33[0].m128i_i64[0] = *(_QWORD *)v9;
    v18 = sub_1E0A0C0(v10);
    v31 = v33[0].m128i_i64[0];
    v33[0].m128i_i64[0] = (unsigned int)sub_15A9FE0(v18, v33[0].m128i_i64[0]);
    v33[0].m128i_i64[0] *= (v33[0].m128i_i64[0] + ((unsigned __int64)(sub_127FA20(v18, v31) + 7) >> 3) - 1)
                         / v33[0].m128i_i64[0];
    v19 = sub_15AAE50(v18, v31);
    v32 = a5[4];
    v20 = sub_1E090F0(*(_QWORD *)(v32 + 56), v33[0].m128i_i64[0], v19, 0, 0, 0);
    v21 = *(_DWORD *)(v18 + 4);
    v33[0].m128i_i32[0] = v20;
    v22 = 8 * sub_15A9520(v18, v21);
    if ( v22 == 32 )
    {
      v23 = 5;
    }
    else if ( v22 > 0x20 )
    {
      v23 = 6;
      if ( v22 != 64 )
      {
        v23 = 7;
        if ( v22 != 128 )
          v23 = v27;
      }
    }
    else
    {
      v23 = 3;
      if ( v22 != 8 )
        v23 = 4 * (v22 == 16);
    }
    v28 = a3;
    v30 = v33[0].m128i_i32[0];
    v24.m128i_i64[0] = (__int64)sub_1D299D0(a5, v33[0].m128i_i32[0], v23, 0, 0);
    v33[0] = v24;
    memset(v38, 0, 24);
    sub_1E341E0((__int64)&v39, v32, v30, 0);
    result = sub_1D2BF40(
               a5,
               a1,
               a2,
               v28,
               a4[31],
               a4[32],
               v33[0].m128i_i64[0],
               v33[0].m128i_i64[1],
               v39,
               v40,
               0,
               0,
               (__int64)v38);
    si128 = _mm_load_si128(v33);
    v35 = v26;
    a4[31] = v33[0].m128i_i64[0];
    v33[1] = si128;
    v34 = result;
    *((_DWORD *)a4 + 64) = si128.m128i_i32[2];
  }
  return result;
}
