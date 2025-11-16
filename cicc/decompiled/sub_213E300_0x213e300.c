// Function: sub_213E300
// Address: 0x213e300
//
_QWORD *__fastcall sub_213E300(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  char *v6; // rdx
  __int64 v7; // rsi
  char v8; // al
  __int64 v9; // rdx
  const void *v10; // r13
  unsigned int v11; // eax
  __int64 v12; // r15
  __int64 v13; // rdx
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // rdx
  _QWORD *v17; // r12
  const void *v19; // [rsp+0h] [rbp-60h]
  __int64 v20; // [rsp+8h] [rbp-58h]
  _BYTE v21[8]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v22; // [rsp+18h] [rbp-48h]
  __int64 v23; // [rsp+20h] [rbp-40h] BYREF
  int v24; // [rsp+28h] [rbp-38h]

  v6 = *(char **)(a2 + 40);
  v7 = *(_QWORD *)(a2 + 72);
  v8 = *v6;
  v9 = *((_QWORD *)v6 + 1);
  v23 = v7;
  v21[0] = v8;
  v22 = v9;
  if ( v7 )
  {
    sub_1623A60((__int64)&v23, v7, 2);
    v8 = v21[0];
  }
  v10 = *(const void **)(a2 + 88);
  v24 = *(_DWORD *)(a2 + 64);
  if ( v8 )
    v11 = word_4310720[(unsigned __int8)(v8 - 14)];
  else
    v11 = sub_1F58D30((__int64)v21);
  v19 = v10;
  v20 = v11;
  v12 = sub_2138AD0(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v14 = v13;
  v15 = sub_2138AD0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL));
  v17 = sub_1D41320(
          *(_QWORD *)(a1 + 8),
          *(unsigned __int8 *)(*(_QWORD *)(v12 + 40) + 16LL * (unsigned int)v14),
          *(const void ***)(*(_QWORD *)(v12 + 40) + 16LL * (unsigned int)v14 + 8),
          (__int64)&v23,
          v12,
          v14,
          a3,
          a4,
          a5,
          v15,
          v16,
          v19,
          v20);
  if ( v23 )
    sub_161E7C0((__int64)&v23, v23);
  return v17;
}
