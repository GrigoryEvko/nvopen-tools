// Function: sub_7FC780
// Address: 0x7fc780
//
__m128i *__fastcall sub_7FC780(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // rbx
  _QWORD *v4; // r13
  __int64 v5; // r12
  _QWORD *v6; // rax
  _QWORD *v7; // r14
  __m128i *v8; // r12
  __int8 v9; // al
  __int64 v10; // r15
  _QWORD *v11; // r14
  __int64 *v12; // r14
  __m128i *v13; // r8
  __int64 v14; // rax
  _BYTE *v15; // rax
  _BYTE *v16; // rax
  _BYTE *v17; // rax
  _QWORD *v18; // r13
  __int64 v20; // [rsp+10h] [rbp-140h]
  _BYTE *v21; // [rsp+18h] [rbp-138h]
  unsigned int v23; // [rsp+3Ch] [rbp-114h] BYREF
  _BYTE v24[32]; // [rsp+40h] [rbp-110h] BYREF
  _BYTE v25[240]; // [rsp+60h] [rbp-F0h] BYREF

  for ( ; *(_BYTE *)(a1 + 140) == 12; a1 = *(_QWORD *)(a1 + 160) )
    ;
  v3 = sub_72D2E0((_QWORD *)a1);
  v4 = sub_72BA30(byte_4F06A51[0]);
  v5 = sub_72CBE0();
  v6 = sub_7259C0(7);
  v6[20] = v5;
  v7 = v6;
  *(_BYTE *)(v6[21] + 16LL) = (2 * (dword_4F06968 == 0)) | *(_BYTE *)(v6[21] + 16LL) & 0xFD;
  if ( v3 )
    *(_QWORD *)v6[21] = sub_724EF0(v3);
  v8 = sub_725FD0();
  v8[10].m128i_i8[12] = 2;
  v9 = v8[5].m128i_i8[8];
  v8[12].m128i_i8[1] |= 0x10u;
  v8[9].m128i_i64[1] = (__int64)v7;
  v8[5].m128i_i8[8] = v9 & 0x8F | 0x10;
  sub_7362F0((__int64)v8, 0);
  v10 = *(_QWORD *)(v8[9].m128i_i64[1] + 168);
  v11 = *(_QWORD **)v10;
  *v11 = sub_724EF0((__int64)v4);
  v12 = sub_7F54F0((__int64)v8, 1, dword_4F07270[0], &v23);
  sub_7F6C60((__int64)v12, v23, (__int64)v25);
  *a2 = (__int64)sub_7E2270(*(_QWORD *)(*(_QWORD *)v10 + 8LL));
  v13 = sub_7E2270((__int64)v4);
  v14 = *a2;
  v20 = (__int64)v13;
  v12[5] = *a2;
  *(_QWORD *)(v14 + 112) = v13;
  sub_7E1740(v12[10], (__int64)v24);
  sub_7E2BA0((__int64)v24);
  sub_7FAFA0((__int64)v24);
  v21 = sub_726B30(5);
  v15 = sub_731250(v20);
  v16 = sub_73DBF0(0x24u, (__int64)v4, (__int64)v15);
  *((_QWORD *)v21 + 6) = sub_7F0830(v16);
  v17 = sub_731250(*a2);
  v18 = sub_73DBF0(0x23u, v3, (__int64)v17);
  sub_7E67B0(v18);
  *((_QWORD *)v21 + 9) = sub_732B10((__int64)v18);
  sub_7E6810((__int64)v21, (__int64)v24, 1);
  sub_7E1780((__int64)v18, a3);
  sub_7FB010((__int64)v12, v23, (__int64)v25);
  return v8;
}
