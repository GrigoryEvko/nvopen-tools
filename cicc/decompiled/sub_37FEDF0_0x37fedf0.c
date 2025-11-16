// Function: sub_37FEDF0
// Address: 0x37fedf0
//
void __fastcall sub_37FEDF0(__int64 a1, __int64 a2, unsigned int *a3, unsigned int *a4, __m128i a5)
{
  __int64 v8; // rsi
  unsigned __int16 *v9; // rax
  int v10; // r9d
  unsigned int v11; // edx
  unsigned __int16 *v12; // rax
  int v13; // r9d
  unsigned __int8 *v14; // rax
  __int64 v15; // rsi
  unsigned int v16; // edx
  __int64 v17; // [rsp+20h] [rbp-40h] BYREF
  int v18; // [rsp+28h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 80);
  v17 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v17, v8, 1);
  v18 = *(_DWORD *)(a2 + 72);
  sub_375E6F0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), (__int64)a3, (__int64)a4);
  v9 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)a3 + 48LL) + 16LL * a3[2]);
  *(_QWORD *)a3 = sub_33FAF80(*(_QWORD *)(a1 + 8), 244, (__int64)&v17, *v9, *((_QWORD *)v9 + 1), v10, a5);
  a3[2] = v11;
  v12 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)a4 + 48LL) + 16LL * a4[2]);
  v14 = sub_33FAF80(*(_QWORD *)(a1 + 8), 244, (__int64)&v17, *v12, *((_QWORD *)v12 + 1), v13, a5);
  v15 = v17;
  *(_QWORD *)a4 = v14;
  a4[2] = v16;
  if ( v15 )
    sub_B91220((__int64)&v17, v15);
}
