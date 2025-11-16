// Function: sub_3829FC0
// Address: 0x3829fc0
//
unsigned __int8 *__fastcall sub_3829FC0(__int64 a1, __int64 a2, __m128i a3)
{
  unsigned __int64 *v4; // rax
  __int64 v5; // rdx
  int v6; // r9d
  __int64 v7; // rsi
  __int64 v8; // r13
  __int64 v9; // r15
  unsigned int v10; // r12d
  unsigned __int8 *v11; // r12
  __int64 v13; // [rsp+0h] [rbp-60h] BYREF
  __int64 v14; // [rsp+8h] [rbp-58h]
  __int64 v15; // [rsp+10h] [rbp-50h] BYREF
  int v16; // [rsp+18h] [rbp-48h]
  __int64 v17; // [rsp+20h] [rbp-40h] BYREF
  int v18; // [rsp+28h] [rbp-38h]

  v4 = *(unsigned __int64 **)(a2 + 40);
  LODWORD(v14) = 0;
  v16 = 0;
  v13 = 0;
  v5 = v4[1];
  v15 = 0;
  sub_375E510(a1, *v4, v5, (__int64)&v13, (__int64)&v15);
  v7 = *(_QWORD *)(a2 + 80);
  v8 = *(_QWORD *)(a1 + 8);
  v9 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
  v10 = **(unsigned __int16 **)(a2 + 48);
  v17 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v17, v7, 1);
  v18 = *(_DWORD *)(a2 + 72);
  v11 = sub_33FAF80(v8, 216, (__int64)&v17, v10, v9, v6, a3);
  if ( v17 )
    sub_B91220((__int64)&v17, v17);
  return v11;
}
