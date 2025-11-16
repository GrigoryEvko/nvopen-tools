// Function: sub_2D36F30
// Address: 0x2d36f30
//
void __fastcall sub_2D36F30(__int64 a1, __int64 a2, _QWORD *a3)
{
  unsigned __int64 v4; // r13
  __int64 v6; // rbx
  __int64 v7; // r9
  __int64 v8; // rax
  __m128i v9; // xmm1
  _QWORD *v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // rax
  _QWORD *v15; // rdx
  __m128i v16; // xmm3
  _QWORD *v17; // rdi
  unsigned int *v18; // rax
  __m128i v19; // xmm5
  _QWORD *v20; // rdi
  unsigned int v21; // r15d
  _QWORD *v22; // rax
  __int64 **v23; // rax
  __int64 v24; // rax
  unsigned int v25; // [rsp+0h] [rbp-F0h]
  _QWORD *v26; // [rsp+0h] [rbp-F0h]
  __int64 v27; // [rsp+8h] [rbp-E8h]
  unsigned __int64 v28; // [rsp+18h] [rbp-D8h] BYREF
  __m128i v29; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v30; // [rsp+30h] [rbp-C0h]
  unsigned __int64 v31; // [rsp+50h] [rbp-A0h]
  __m128i v32; // [rsp+60h] [rbp-90h] BYREF
  __m128i v33; // [rsp+70h] [rbp-80h] BYREF
  __int64 v34; // [rsp+80h] [rbp-70h]
  __m128i v35; // [rsp+90h] [rbp-60h] BYREF
  __m128i v36; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v37; // [rsp+B0h] [rbp-40h]

  v4 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v6 = *(_QWORD *)(a1 + 136);
  if ( (a2 & 4) != 0 )
  {
    sub_AF48C0(&v35, v4);
    v32.m128i_i64[0] = v35.m128i_i64[0];
    v32.m128i_i64[1] = v37;
    if ( sub_2D2BDF0(v6, v32.m128i_i64) )
    {
      sub_AF48C0(&v32, v4);
      v19 = _mm_loadu_si128(&v33);
      v20 = *(_QWORD **)(a1 + 144);
      v35 = _mm_loadu_si128(&v32);
      v36 = v19;
      v37 = v34;
      v21 = sub_2D2C1F0(v20, &v35);
      v35 = (__m128i)1uLL;
      v36.m128i_i64[0] = 0;
      sub_2D23C40(a1, a3, v21, &v35);
      sub_2D301F0(a1, a3, v21, 1);
      sub_2D367C0(a1, 1, v4, v4 | 4);
    }
  }
  else
  {
    v7 = sub_B10D40(v4 + 48);
    v8 = *(_QWORD *)(*(_QWORD *)(v4 + 32 * (1LL - (*(_DWORD *)(v4 + 4) & 0x7FFFFFF))) + 24LL);
    v35.m128i_i64[1] = v7;
    v35.m128i_i64[0] = v8;
    if ( sub_2D2BDF0(v6, v35.m128i_i64) )
    {
      sub_AF4850((__int64)&v32, v4);
      v9 = _mm_loadu_si128(&v33);
      v10 = *(_QWORD **)(a1 + 144);
      v35 = _mm_loadu_si128(&v32);
      v36 = v9;
      v37 = v34;
      v25 = sub_2D2C1F0(v10, &v35);
      v29 = (__m128i)1uLL;
      v30 = 0;
      sub_2D23C40(a1, a3, v25, &v29);
      v11 = (__int64)a3;
      sub_2D301F0(a1, a3, v25, 1);
      v12 = sub_B10CD0(v4 + 48);
      v31 = v4;
      v13 = v12;
      v14 = *(_DWORD *)(v4 + 4) & 0x7FFFFFF;
      v15 = *(_QWORD **)(*(_QWORD *)(v4 - 32 * v14) + 24LL);
      v27 = *(_QWORD *)(*(_QWORD *)(v4 + 32 * (2 - v14)) + 24LL);
      if ( !v15 )
      {
        v22 = (_QWORD *)sub_BD5C60(v4);
        v23 = (__int64 **)sub_BCB2A0(v22);
        v24 = sub_ACADE0(v23);
        v15 = sub_B98A20(v24, v11);
      }
      v26 = v15;
      v28 = sub_2D28460(v31);
      sub_AF4850((__int64)&v32, v4);
      v16 = _mm_loadu_si128(&v33);
      v17 = *(_QWORD **)(a1 + 144);
      v35 = _mm_loadu_si128(&v32);
      v36 = v16;
      v37 = v34;
      v35.m128i_i32[0] = sub_2D2C1F0(v17, &v35);
      v36.m128i_i64[1] = (__int64)v26;
      v35.m128i_i64[1] = v27;
      v36.m128i_i64[0] = 0;
      sub_B10CB0(&v32, v13);
      sub_9C6650(&v36);
      v36.m128i_i64[0] = v32.m128i_i64[0];
      if ( v32.m128i_i64[0] )
      {
        sub_B976B0((__int64)&v32, (unsigned __int8 *)v32.m128i_i64[0], (__int64)&v36);
        v32.m128i_i64[0] = 0;
      }
      sub_9C6650(&v32);
      v18 = (unsigned int *)sub_2D363E0(a1 + 72, (__int64 *)&v28);
      sub_2D29B40(v18, (unsigned __int64)&v35);
      sub_9C6650(&v36);
    }
  }
}
