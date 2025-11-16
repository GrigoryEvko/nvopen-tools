// Function: sub_2139850
// Address: 0x2139850
//
__int64 __fastcall sub_2139850(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 *v6; // rax
  __int64 v7; // rsi
  __int64 *v8; // r14
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 *v11; // r12
  unsigned __int8 *v12; // rdx
  __int64 v13; // rcx
  const void **v14; // r8
  __int64 v15; // r12
  __int128 v17; // [rsp-10h] [rbp-60h]
  __int64 v18; // [rsp+0h] [rbp-50h]
  const void **v19; // [rsp+8h] [rbp-48h]
  __int64 v20; // [rsp+10h] [rbp-40h] BYREF
  int v21; // [rsp+18h] [rbp-38h]

  v6 = sub_2139210(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), a3, a4, a5);
  v7 = *(_QWORD *)(a2 + 72);
  v8 = *(__int64 **)(a1 + 8);
  v10 = v9;
  v11 = v6;
  v12 = (unsigned __int8 *)(v6[5] + 16LL * (unsigned int)v9);
  v13 = *v12;
  v14 = (const void **)*((_QWORD *)v12 + 1);
  v20 = v7;
  if ( v7 )
  {
    v18 = v13;
    v19 = v14;
    sub_1623A60((__int64)&v20, v7, 2);
    v13 = v18;
    v14 = v19;
  }
  *((_QWORD *)&v17 + 1) = v10;
  *(_QWORD *)&v17 = v11;
  v21 = *(_DWORD *)(a2 + 64);
  v15 = sub_1D309E0(v8, 130, (__int64)&v20, v13, v14, 0, *(double *)a3.m128i_i64, a4, *(double *)a5.m128i_i64, v17);
  if ( v20 )
    sub_161E7C0((__int64)&v20, v20);
  return v15;
}
