// Function: sub_3423F00
// Address: 0x3423f00
//
void __fastcall __noreturn sub_3423F00(__int64 a1, __int64 a2)
{
  __m128i *v2; // rdx
  __m128i si128; // xmm0
  _QWORD v4[2]; // [rsp+0h] [rbp-C0h] BYREF
  char v5; // [rsp+10h] [rbp-B0h] BYREF
  unsigned __int8 *v6[2]; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v7; // [rsp+30h] [rbp-90h] BYREF
  __int16 v8; // [rsp+40h] [rbp-80h]
  _QWORD v9[3]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v10; // [rsp+68h] [rbp-58h]
  __m128i *v11; // [rsp+70h] [rbp-50h]
  __int64 v12; // [rsp+78h] [rbp-48h]
  _QWORD *v13; // [rsp+80h] [rbp-40h]

  v4[0] = &v5;
  v12 = 0x100000000LL;
  v4[1] = 0;
  v5 = 0;
  v9[0] = &unk_49DD210;
  v9[1] = 0;
  v9[2] = 0;
  v10 = 0;
  v11 = 0;
  v13 = v4;
  sub_CB5980((__int64)v9, 0, 0, 0);
  sub_2C75F20((__int64)v6, (__int64 *)(a2 + 80));
  sub_CB6200((__int64)v9, v6[0], (size_t)v6[1]);
  if ( (__int64 *)v6[0] != &v7 )
    j_j___libc_free_0((unsigned __int64)v6[0]);
  v2 = v11;
  if ( (unsigned __int64)(v10 - (_QWORD)v11) <= 0x1C )
  {
    sub_CB6200((__int64)v9, " Error: unsupported operation", 0x1Du);
  }
  else
  {
    si128 = _mm_load_si128(xmmword_44E1280);
    v11[1].m128i_i32[2] = 1869182049;
    v2[1].m128i_i64[0] = 0x7265706F20646574LL;
    v2[1].m128i_i8[12] = 110;
    *v2 = si128;
    v11 = (__m128i *)((char *)v11 + 29);
  }
  v6[0] = (unsigned __int8 *)v4;
  v8 = 260;
  sub_C64D30((__int64)v6, 1u);
}
