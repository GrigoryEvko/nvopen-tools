// Function: sub_2D36AC0
// Address: 0x2d36ac0
//
void __fastcall sub_2D36AC0(__int64 *a1, _QWORD *a2, __int64 a3)
{
  _QWORD *v4; // r12
  unsigned __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // rsi
  _QWORD *v9; // rdi
  __m128i v10; // xmm0
  __int32 v11; // eax
  __int64 v12; // rsi
  unsigned int *v13; // rax
  _QWORD *v14; // rax
  __int64 **v15; // rax
  __int64 v16; // rax
  unsigned __int64 v17; // [rsp+8h] [rbp-98h] BYREF
  __m128i v18; // [rsp+10h] [rbp-90h] BYREF
  __m128i v19; // [rsp+20h] [rbp-80h] BYREF
  __int64 v20; // [rsp+30h] [rbp-70h]
  __m128i v21; // [rsp+40h] [rbp-60h] BYREF
  __m128i v22; // [rsp+50h] [rbp-50h] BYREF
  __int64 v23; // [rsp+60h] [rbp-40h]

  v4 = a2;
  if ( !a2 )
  {
    v14 = (_QWORD *)sub_BD5C60(a1[1]);
    v15 = (__int64 **)sub_BCB2A0(v14);
    v16 = sub_ACADE0(v15);
    v4 = sub_B98A20(v16, 0);
  }
  v6 = sub_2D28460(a1[2]);
  v7 = *a1;
  v8 = a1[1];
  v17 = v6;
  sub_AF4850((__int64)&v18, v8);
  v9 = *(_QWORD **)(v7 + 144);
  v10 = _mm_loadu_si128(&v18);
  v22 = _mm_loadu_si128(&v19);
  v21 = v10;
  v23 = v20;
  v11 = sub_2D2C1F0(v9, &v21);
  v12 = a1[3];
  v21.m128i_i64[1] = a3;
  v22.m128i_i64[0] = 0;
  v21.m128i_i32[0] = v11;
  v22.m128i_i64[1] = (__int64)v4;
  sub_B10CB0(&v18, v12);
  if ( v22.m128i_i64[0] )
    sub_B91220((__int64)&v22, v22.m128i_i64[0]);
  v22.m128i_i64[0] = v18.m128i_i64[0];
  if ( v18.m128i_i64[0] )
    sub_B976B0((__int64)&v18, (unsigned __int8 *)v18.m128i_i64[0], (__int64)&v22);
  v13 = (unsigned int *)sub_2D363E0(*a1 + 72, (__int64 *)&v17);
  sub_2D29B40(v13, (unsigned __int64)&v21);
  if ( v22.m128i_i64[0] )
    sub_B91220((__int64)&v22, v22.m128i_i64[0]);
}
