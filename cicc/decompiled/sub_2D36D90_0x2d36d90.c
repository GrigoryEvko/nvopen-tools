// Function: sub_2D36D90
// Address: 0x2d36d90
//
void __fastcall sub_2D36D90(__int64 *a1, __int64 a2)
{
  __int64 v4; // r13
  __int64 v5; // r8
  __int64 v6; // rax
  __int64 v7; // r13
  __m128i v8; // xmm1
  _QWORD *v9; // rdi
  unsigned int v10; // eax
  __int64 v11; // rdi
  unsigned int v12; // r13d
  __int64 v13; // rax
  __int64 v14; // r15
  _QWORD *v15; // rsi
  int v16; // r14d
  __m128i v17; // [rsp+10h] [rbp-90h] BYREF
  __m128i v18; // [rsp+20h] [rbp-80h] BYREF
  __int64 v19; // [rsp+30h] [rbp-70h]
  __m128i v20; // [rsp+40h] [rbp-60h] BYREF
  __m128i v21; // [rsp+50h] [rbp-50h]
  __int64 v22; // [rsp+60h] [rbp-40h]

  v4 = *(_QWORD *)(*a1 + 136);
  v5 = sub_B10D40(a2 + 48);
  v6 = *(_QWORD *)(*(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) + 24LL);
  v20.m128i_i64[1] = v5;
  v20.m128i_i64[0] = v6;
  if ( sub_2D2BDF0(v4, v20.m128i_i64) )
  {
    v7 = *a1;
    sub_AF4850((__int64)&v17, a2);
    v8 = _mm_loadu_si128(&v18);
    v9 = *(_QWORD **)(v7 + 144);
    v20 = _mm_loadu_si128(&v17);
    v21 = v8;
    v22 = v19;
    v10 = sub_2D2C1F0(v9, &v20);
    v11 = *a1;
    v12 = v10;
    v13 = *(_QWORD *)(*(_QWORD *)(a2 + 32 * (3LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) + 24LL);
    v20.m128i_i32[0] = 0;
    v21.m128i_i64[0] = a2;
    v20.m128i_i64[1] = v13;
    sub_2D23C40(v11, *(_QWORD **)a1[1], v12, &v20);
    v14 = *a1;
    v15 = *(_QWORD **)a1[1];
    if ( (unsigned __int8)sub_2D23AC0(*a1, v15, 0, v12, v20.m128i_i32) )
    {
      v16 = (unsigned __int8)sub_B59AF0(a2);
      sub_2D301F0(*a1, *(_QWORD **)a1[1], v12, v16);
      sub_2D36C00(*a1, v16, a2, a2);
    }
    else
    {
      sub_2D301F0(v14, v15, v12, 1);
      sub_2D36C00(*a1, 1, a2, a2);
    }
  }
}
