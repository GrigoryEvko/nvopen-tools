// Function: sub_2D36940
// Address: 0x2d36940
//
void __fastcall sub_2D36940(__int64 *a1, __int64 a2)
{
  __int64 v4; // r15
  __int64 v5; // r15
  __m128i v6; // xmm1
  _QWORD *v7; // rdi
  unsigned int v8; // r13d
  __int64 v9; // rax
  __int64 v10; // rdi
  _QWORD **v11; // rax
  __int64 v12; // rdi
  _QWORD *v13; // rsi
  int v14; // r14d
  __m128i v15; // [rsp+10h] [rbp-90h] BYREF
  __m128i v16; // [rsp+20h] [rbp-80h] BYREF
  __int64 v17; // [rsp+30h] [rbp-70h]
  __m128i v18; // [rsp+40h] [rbp-60h] BYREF
  __m128i v19; // [rsp+50h] [rbp-50h]
  __int64 v20; // [rsp+60h] [rbp-40h]

  v4 = *(_QWORD *)(*a1 + 136);
  sub_AF48C0(&v18, a2);
  v15.m128i_i64[0] = v18.m128i_i64[0];
  v15.m128i_i64[1] = v20;
  if ( sub_2D2BDF0(v4, v15.m128i_i64) )
  {
    v5 = *a1;
    sub_AF48C0(&v15, a2);
    v6 = _mm_loadu_si128(&v16);
    v7 = *(_QWORD **)(v5 + 144);
    v18 = _mm_loadu_si128(&v15);
    v19 = v6;
    v20 = v17;
    v8 = sub_2D2C1F0(v7, &v18);
    v9 = sub_B13870(a2);
    v19.m128i_i64[0] = a2 | 4;
    v10 = *a1;
    v18.m128i_i64[1] = v9;
    v11 = (_QWORD **)a1[1];
    v18.m128i_i32[0] = 0;
    sub_2D23C40(v10, *v11, v8, &v18);
    v12 = *a1;
    v13 = *(_QWORD **)a1[1];
    if ( (unsigned __int8)sub_2D23AC0(*a1, v13, 0, v8, v18.m128i_i32) )
    {
      v14 = (unsigned __int8)sub_B14070(a2);
      sub_2D301F0(*a1, *(_QWORD **)a1[1], v8, v14);
      sub_2D367C0(*a1, v14, a2, a2 | 4);
    }
    else
    {
      sub_2D301F0(v12, v13, v8, 1);
      sub_2D367C0(*a1, 1, a2, a2 | 4);
    }
  }
}
