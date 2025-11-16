// Function: sub_C8D440
// Address: 0xc8d440
//
void __fastcall __noreturn sub_C8D440(unsigned __int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  __int64 v4; // rdx
  unsigned __int64 v5; // rcx
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rcx
  __m128i *v9; // rax
  __int64 v10; // rcx
  __m128i *v11; // rdx
  __m128i v12[2]; // [rsp+10h] [rbp-100h] BYREF
  _BYTE v13[32]; // [rsp+30h] [rbp-E0h] BYREF
  __m128i v14[2]; // [rsp+50h] [rbp-C0h] BYREF
  __m128i v15; // [rsp+70h] [rbp-A0h] BYREF
  unsigned __int64 v16; // [rsp+80h] [rbp-90h] BYREF
  unsigned __int64 *v17; // [rsp+90h] [rbp-80h] BYREF
  __int64 v18; // [rsp+98h] [rbp-78h]
  unsigned __int64 v19; // [rsp+A0h] [rbp-70h] BYREF
  _QWORD v20[2]; // [rsp+B0h] [rbp-60h] BYREF
  __m128i v21; // [rsp+C0h] [rbp-50h] BYREF
  __int16 v22; // [rsp+D0h] [rbp-40h]

  sub_BAE5F0((__int64)&v17, 0xFFFFFFFF, a3, a4);
  sub_BAE5F0((__int64)v13, a1, v4, v5);
  sub_95D570(v14, "SmallVector unable to grow. Requested capacity (", (__int64)v13);
  sub_94F930(&v15, (__int64)v14, ") is larger than maximum value for size type (");
  v6 = 15;
  v7 = 15;
  if ( (unsigned __int64 *)v15.m128i_i64[0] != &v16 )
    v7 = v16;
  v8 = v15.m128i_i64[1] + v18;
  if ( v15.m128i_i64[1] + v18 <= v7 )
    goto LABEL_7;
  if ( v17 != &v19 )
    v6 = v19;
  if ( v8 <= v6 )
  {
    v9 = (__m128i *)sub_2241130(&v17, 0, 0, v15.m128i_i64[0], v15.m128i_i64[1]);
    v20[0] = &v21;
    v10 = v9->m128i_i64[0];
    v11 = v9 + 1;
    if ( (__m128i *)v9->m128i_i64[0] != &v9[1] )
      goto LABEL_8;
  }
  else
  {
LABEL_7:
    v9 = (__m128i *)sub_2241490(&v15, v17, v18, v8);
    v20[0] = &v21;
    v10 = v9->m128i_i64[0];
    v11 = v9 + 1;
    if ( (__m128i *)v9->m128i_i64[0] != &v9[1] )
    {
LABEL_8:
      v20[0] = v10;
      v21.m128i_i64[0] = v9[1].m128i_i64[0];
      goto LABEL_9;
    }
  }
  v21 = _mm_loadu_si128(v9 + 1);
LABEL_9:
  v20[1] = v9->m128i_i64[1];
  v9->m128i_i64[0] = (__int64)v11;
  v9->m128i_i64[1] = 0;
  v9[1].m128i_i8[0] = 0;
  sub_94F930(v12, (__int64)v20, ")");
  sub_2240A30(v20);
  sub_2240A30(&v15);
  sub_2240A30(v14);
  sub_2240A30(v13);
  sub_2240A30(&v17);
  v22 = 260;
  v20[0] = v12;
  sub_C64D30((__int64)v20, 1u);
}
