// Function: sub_122C010
// Address: 0x122c010
//
__int64 __fastcall sub_122C010(__int64 a1, __int64 a2, __int64 a3, __m128i *a4)
{
  __int64 v6; // rdi
  unsigned int v7; // r12d
  unsigned __int64 v8; // rsi
  int v11; // eax
  __int64 v12; // rax
  __m128i v13; // xmm3
  __m128i v14; // xmm2
  __int64 v15; // rax
  __m128i v16; // [rsp+0h] [rbp-90h] BYREF
  __int64 v17; // [rsp+10h] [rbp-80h]
  __int64 v18; // [rsp+18h] [rbp-78h]
  __int16 v19; // [rsp+20h] [rbp-70h]
  __m128i v20; // [rsp+30h] [rbp-60h] BYREF
  __m128i v21; // [rsp+40h] [rbp-50h]
  __int16 v22; // [rsp+50h] [rbp-40h]

  v6 = a1 + 176;
  v7 = a4[3].m128i_u8[0];
  if ( (_BYTE)v7 )
  {
    v17 = a2;
    v8 = *(_QWORD *)(a1 + 232);
    v19 = 1283;
    v16.m128i_i64[0] = (__int64)"field '";
    v18 = a3;
    v20.m128i_i64[0] = (__int64)&v16;
    v22 = 770;
    v21.m128i_i64[0] = (__int64)"' cannot be specified more than once";
    sub_11FD800(v6, v8, (__int64)&v20, 1);
  }
  else
  {
    v11 = sub_1205200(v6);
    *(_DWORD *)(a1 + 240) = v11;
    if ( v11 == 529 )
    {
      v14 = _mm_loadu_si128(a4 + 1);
      v20 = _mm_loadu_si128(a4);
      v21 = v14;
      v7 = sub_1208D50(a1, a2, a3, (__int64)&v20);
      if ( !(_BYTE)v7 )
      {
        v15 = v20.m128i_i64[0];
        a4[3].m128i_i8[0] = 1;
        a4[3].m128i_i32[1] = 1;
        a4->m128i_i64[0] = v15;
        a4->m128i_i8[8] = v20.m128i_i8[8];
        a4[1] = v21;
      }
    }
    else
    {
      v16 = _mm_loadu_si128(a4 + 2);
      v7 = sub_1225CE0(a1, a2, a3, (__int64)&v16);
      if ( !(_BYTE)v7 )
      {
        v12 = v16.m128i_i64[0];
        v13 = _mm_loadu_si128(&v16);
        a4[3].m128i_i8[0] = 1;
        a4[3].m128i_i32[1] = 2;
        a4[2].m128i_i64[0] = v12;
        v20 = v13;
        a4[2].m128i_i16[4] = v13.m128i_i16[4];
      }
    }
  }
  return v7;
}
