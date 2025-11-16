// Function: sub_381E8A0
// Address: 0x381e8a0
//
void __fastcall sub_381E8A0(__int64 a1, __int64 a2, unsigned int *a3, __int64 a4, __m128i a5)
{
  __int64 v8; // rsi
  unsigned __int16 *v9; // rax
  unsigned __int16 v10; // r15
  __int64 v11; // r9
  int v12; // r9d
  unsigned int v13; // edx
  unsigned __int8 *v14; // rax
  __int64 v15; // rsi
  int v16; // edx
  _QWORD *v17; // [rsp+10h] [rbp-70h]
  __int64 v18; // [rsp+18h] [rbp-68h]
  __int64 v19; // [rsp+40h] [rbp-40h] BYREF
  int v20; // [rsp+48h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 80);
  v19 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v19, v8, 1);
  v20 = *(_DWORD *)(a2 + 72);
  sub_375E510(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), (__int64)a3, a4);
  v9 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)a3 + 48LL) + 16LL * a3[2]);
  v10 = *v9;
  v18 = *((_QWORD *)v9 + 1);
  v17 = *(_QWORD **)(a1 + 8);
  sub_3406EB0(v17, 0xBCu, (__int64)&v19, *v9, v18, v11, *(_OWORD *)a3, *(_OWORD *)a4);
  *(_QWORD *)a3 = sub_33FAF80((__int64)v17, 202, (__int64)&v19, v10, v18, v12, a5);
  a3[2] = v13;
  v14 = sub_3400BD0(*(_QWORD *)(a1 + 8), 0, (__int64)&v19, v10, v18, 0, a5, 0);
  v15 = v19;
  *(_QWORD *)a4 = v14;
  *(_DWORD *)(a4 + 8) = v16;
  if ( v15 )
    sub_B91220((__int64)&v19, v15);
}
