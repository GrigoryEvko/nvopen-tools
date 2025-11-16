// Function: sub_2527D70
// Address: 0x2527d70
//
__int64 __fastcall sub_2527D70(__int64 *a1, _BYTE *a2)
{
  __int64 v3; // r13
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r12
  unsigned __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v10; // r13
  _BYTE *v11; // r15
  __m128i v12; // rax
  __m128i v13; // rax
  unsigned __int64 v14; // rdi
  __int64 v15; // r13
  unsigned __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r12
  unsigned __int64 v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // r13
  _BYTE *v22; // r14
  __m128i v23; // rax
  __int64 v24; // rdx
  __m128i v25; // [rsp+10h] [rbp-50h] BYREF
  __m128i v26[4]; // [rsp+20h] [rbp-40h] BYREF

  v3 = *a1;
  if ( *a2 == 61 )
  {
    v4 = sub_250D2C0(*((_QWORD *)a2 - 4), 0);
    sub_2521E40(v3, v4, v5, 0, 2, 0, 1);
    if ( (_BYTE)qword_4FEE648 )
    {
      v21 = *a1;
      v22 = (_BYTE *)a1[1];
      v23.m128i_i64[0] = sub_250D2C0((unsigned __int64)a2, 0);
      v25 = v23;
      v26[0].m128i_i64[0] = sub_2527850(v21, &v25, 0, v22, 1u);
      v26[0].m128i_i64[1] = v24;
    }
    v6 = *a1;
    v7 = sub_250D2C0(*((_QWORD *)a2 - 4), 0);
    sub_2521100(v6, v7, v8, 0, 2, 0, 1);
  }
  else
  {
    sub_250D230((unsigned __int64 *)v26, (unsigned __int64)a2, 1, 0);
    sub_251BBC0(v3, v26[0].m128i_i64[0], v26[0].m128i_i64[1], 0, 2, 0, 1);
    v10 = *a1;
    v11 = (_BYTE *)a1[1];
    v12.m128i_i64[0] = sub_250D2C0(*((_QWORD *)a2 - 8), 0);
    v26[0] = v12;
    v13.m128i_i64[0] = sub_2527850(v10, v26, 0, v11, 1u);
    v14 = *((_QWORD *)a2 - 4);
    v15 = *a1;
    v25 = v13;
    v16 = sub_250D2C0(v14, 0);
    sub_2521E40(v15, v16, v17, 0, 2, 0, 1);
    v18 = *a1;
    v19 = sub_250D2C0(*((_QWORD *)a2 - 4), 0);
    sub_2521100(v18, v19, v20, 0, 2, 0, 1);
  }
  return 1;
}
