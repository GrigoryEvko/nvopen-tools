// Function: sub_8C2140
// Address: 0x8c2140
//
_QWORD *__fastcall sub_8C2140(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // r15
  __m128i *v5; // r14
  _QWORD *v6; // rax
  _QWORD *v7; // r13
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // rax
  __m128i *v12; // rdi
  _QWORD *v13; // rax
  _QWORD *v14; // r12
  __m128i v17[7]; // [rsp+10h] [rbp-70h] BYREF

  sub_878710(a1, v17);
  sub_87A680(v17, (__int64 *)(a1 + 48), 0);
  v4 = sub_87EBB0(0xAu, v17[0].m128i_i64[0], (_QWORD *)(a1 + 48));
  *((_DWORD *)v4 + 10) = *(_DWORD *)(a1 + 40);
  v5 = sub_725FD0();
  v6 = sub_7259C0(7);
  v5[9].m128i_i64[1] = (__int64)v6;
  v7 = v6;
  v8 = v6[21];
  v9 = sub_72CBE0();
  v10 = a2;
  v7[20] = v9;
  if ( a3 )
  {
    *(_QWORD *)v8 = a3;
    sub_8DCB20(v7);
    v10 = a2;
  }
  *(_WORD *)(v8 + 16) |= 0x102u;
  *(_BYTE *)(v8 + 21) |= 1u;
  *(_QWORD *)(v8 + 40) = v10;
  v5[10].m128i_i8[14] = 1;
  v4[11] = v5;
  v11 = *(_QWORD *)(a1 + 88);
  v12 = *(__m128i **)(v11 + 88);
  if ( v12 )
  {
    if ( (*(_BYTE *)(v11 + 160) & 1) != 0 )
      v12 = (__m128i *)a1;
  }
  else
  {
    v12 = (__m128i *)a1;
  }
  v13 = sub_8C18E0(v12, v10, (__int64)v4, a1);
  v14 = v13;
  if ( v13 )
    sub_87F0B0((__int64)v13, (__int64 *)(*(_QWORD *)(a1 + 88) + 216LL));
  return v14;
}
