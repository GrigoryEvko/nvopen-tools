// Function: sub_213BF00
// Address: 0x213bf00
//
__int64 *__fastcall sub_213BF00(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 *v8; // r14
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // r13
  __int64 v11; // r9
  __int64 v12; // r12
  unsigned __int8 *v13; // rdx
  __int64 v14; // rcx
  const void **v15; // r8
  __int64 *v16; // r12
  __int64 v18; // [rsp+8h] [rbp-58h]
  __int64 v19; // [rsp+10h] [rbp-50h]
  const void **v20; // [rsp+18h] [rbp-48h]
  __int64 v21; // [rsp+20h] [rbp-40h] BYREF
  int v22; // [rsp+28h] [rbp-38h]

  v6 = sub_2138AD0(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v7 = *(_QWORD *)(a2 + 72);
  v8 = *(__int64 **)(a1 + 8);
  v10 = v9;
  v11 = *(_QWORD *)(a2 + 32);
  v12 = v6;
  v13 = (unsigned __int8 *)(*(_QWORD *)(v6 + 40) + 16LL * (unsigned int)v9);
  v14 = *v13;
  v15 = (const void **)*((_QWORD *)v13 + 1);
  v21 = v7;
  if ( v7 )
  {
    v18 = v14;
    v19 = v11;
    v20 = v15;
    sub_1623A60((__int64)&v21, v7, 2);
    v14 = v18;
    v11 = v19;
    v15 = v20;
  }
  v22 = *(_DWORD *)(a2 + 64);
  v16 = sub_1D332F0(v8, 148, (__int64)&v21, v14, v15, 0, a3, a4, a5, v12, v10, *(_OWORD *)(v11 + 40));
  if ( v21 )
    sub_161E7C0((__int64)&v21, v21);
  return v16;
}
