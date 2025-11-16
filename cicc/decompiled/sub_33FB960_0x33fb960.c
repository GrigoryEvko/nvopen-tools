// Function: sub_33FB960
// Address: 0x33fb960
//
unsigned __int8 *__fastcall sub_33FB960(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __m128i a4,
        __int64 a5,
        __int64 a6,
        int a7)
{
  unsigned __int16 *v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // r8
  unsigned __int8 *v12; // r12
  __int64 v14; // [rsp+0h] [rbp-50h]
  __int64 v15; // [rsp+8h] [rbp-48h]
  __int64 v16; // [rsp+10h] [rbp-40h] BYREF
  int v17; // [rsp+18h] [rbp-38h]

  v8 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * a3);
  v9 = *(_QWORD *)(a2 + 80);
  v10 = *v8;
  v11 = *((_QWORD *)v8 + 1);
  v16 = v9;
  if ( v9 )
  {
    v14 = v10;
    v15 = v11;
    sub_B96E90((__int64)&v16, v9, 1);
    v10 = v14;
    v11 = v15;
  }
  v17 = *(_DWORD *)(a2 + 72);
  v12 = sub_33FAF80(a1, 52, (__int64)&v16, v10, v11, a7, a4);
  if ( v16 )
    sub_B91220((__int64)&v16, v16);
  return v12;
}
