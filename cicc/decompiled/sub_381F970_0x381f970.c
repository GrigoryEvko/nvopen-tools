// Function: sub_381F970
// Address: 0x381f970
//
void __fastcall sub_381F970(__int64 a1, __int64 a2, unsigned int *a3, __int64 a4, __m128i a5)
{
  __int64 v7; // rsi
  unsigned __int16 *v8; // rax
  __int64 v9; // r14
  unsigned __int16 v10; // r15
  __int128 v11; // rax
  int v12; // r9d
  __int128 v13; // rax
  unsigned int v14; // edx
  unsigned __int8 *v15; // rax
  __int64 v16; // rsi
  int v17; // edx
  __int128 v18; // [rsp+0h] [rbp-80h]
  _QWORD *v19; // [rsp+10h] [rbp-70h]
  __int64 v21; // [rsp+40h] [rbp-40h] BYREF
  int v22; // [rsp+48h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 80);
  v21 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v21, v7, 1);
  v22 = *(_DWORD *)(a2 + 72);
  sub_375E510(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), (__int64)a3, a4);
  v8 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)a3 + 48LL) + 16LL * a3[2]);
  v9 = *((_QWORD *)v8 + 1);
  v10 = *v8;
  v19 = *(_QWORD **)(a1 + 8);
  *(_QWORD *)&v11 = sub_33FAF80((__int64)v19, 200, (__int64)&v21, *v8, v9, (_DWORD)v19, a5);
  v18 = v11;
  *(_QWORD *)&v13 = sub_33FAF80(*(_QWORD *)(a1 + 8), 200, (__int64)&v21, v10, v9, v12, a5);
  *(_QWORD *)a3 = sub_3406EB0(v19, 0x38u, (__int64)&v21, v10, v9, (__int64)v19, v13, v18);
  a3[2] = v14;
  v15 = sub_3400BD0(*(_QWORD *)(a1 + 8), 0, (__int64)&v21, v10, v9, 0, a5, 0);
  v16 = v21;
  *(_QWORD *)a4 = v15;
  *(_DWORD *)(a4 + 8) = v17;
  if ( v16 )
    sub_B91220((__int64)&v21, v16);
}
