// Function: sub_2177B20
// Address: 0x2177b20
//
__int64 *__fastcall sub_2177B20(__int64 a1, double a2, double a3, __m128i a4, __int64 a5, __int64 *a6, char a7)
{
  __int64 *v8; // rax
  __int64 v9; // rsi
  __int64 v10; // r13
  int v11; // r14d
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rbx
  __int64 v17; // rsi
  unsigned int v18; // edx
  unsigned __int64 v19; // rbx
  __int64 *v20; // r12
  unsigned __int64 v22; // rdx
  __int128 v23; // [rsp-10h] [rbp-A0h]
  char v24; // [rsp+Ch] [rbp-84h]
  __int64 v25; // [rsp+10h] [rbp-80h]
  __int64 v26; // [rsp+18h] [rbp-78h]
  __int64 v27; // [rsp+30h] [rbp-60h] BYREF
  int v28; // [rsp+38h] [rbp-58h]
  __int64 v29; // [rsp+40h] [rbp-50h] BYREF
  int v30; // [rsp+48h] [rbp-48h]
  __int64 v31; // [rsp+50h] [rbp-40h]
  unsigned __int64 v32; // [rsp+58h] [rbp-38h]

  v8 = *(__int64 **)(a1 + 32);
  v9 = *(_QWORD *)(a1 + 72);
  v10 = *v8;
  v11 = *((_DWORD *)v8 + 2);
  v27 = v9;
  v12 = v8[5];
  v13 = v8[6];
  if ( v9 )
  {
    v24 = a7;
    v25 = v8[5];
    v26 = v8[6];
    sub_1623A60((__int64)&v27, v9, 2);
    a7 = v24;
    v12 = v25;
    v13 = v26;
  }
  v28 = *(_DWORD *)(a1 + 64);
  if ( a7 )
  {
    v14 = sub_1D2B130(a6, (__int64)&v27, 6u, 0, v12, v13, 0, 5u);
    v16 = v15;
    v17 = sub_1D323C0(a6, (__int64)v14, v15, (__int64)&v27, 5, 0, a2, a3, *(double *)a4.m128i_i64);
    v19 = v18 | v16 & 0xFFFFFFFF00000000LL;
  }
  else
  {
    v17 = (__int64)sub_1D2B130(a6, (__int64)&v27, 5u, 0, v12, v13, 0, 5u);
    v19 = v22;
  }
  v31 = v17;
  *((_QWORD *)&v23 + 1) = 2;
  *(_QWORD *)&v23 = &v29;
  v29 = v10;
  v30 = v11;
  v32 = v19;
  v20 = sub_1D359D0(a6, 304, (__int64)&v27, 1, 0, 0, a2, a3, a4, v23);
  if ( v27 )
    sub_161E7C0((__int64)&v27, v27);
  return v20;
}
