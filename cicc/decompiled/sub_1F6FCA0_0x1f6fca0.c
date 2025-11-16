// Function: sub_1F6FCA0
// Address: 0x1f6fca0
//
__int64 __fastcall sub_1F6FCA0(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 *v7; // rax
  __int64 v8; // r15
  __int64 v9; // rax
  const void **v10; // r10
  unsigned int v11; // r13d
  const void **v12; // r8
  __int64 v13; // rdx
  int v14; // eax
  __int64 v15; // rcx
  __int64 result; // rax
  __int64 v17; // rsi
  __int64 v18; // rsi
  __int64 *v19; // rdi
  __int128 v20; // rax
  __int64 v21; // [rsp+8h] [rbp-58h]
  const void **v22; // [rsp+10h] [rbp-50h]
  const void **v23; // [rsp+18h] [rbp-48h]
  const void **v24; // [rsp+18h] [rbp-48h]
  __int64 v25; // [rsp+18h] [rbp-48h]
  __int64 v26; // [rsp+20h] [rbp-40h] BYREF
  int v27; // [rsp+28h] [rbp-38h]

  v7 = *(__int64 **)(a2 + 32);
  v8 = *v7;
  v9 = v7[5];
  v10 = *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL);
  v11 = **(unsigned __int8 **)(a2 + 40);
  v12 = *(const void ***)(v9 + 96);
  v13 = *(unsigned __int8 *)(v9 + 88);
  v14 = *(unsigned __int16 *)(v8 + 24);
  v15 = (unsigned __int8)v13;
  if ( v14 != 33 && v14 != 11
    || *(_BYTE *)(a1 + 25) && (!(_BYTE)v13 || !*(_QWORD *)(*(_QWORD *)(a1 + 8) + 8 * v13 + 120)) )
  {
    return 0;
  }
  v17 = *(_QWORD *)(a2 + 72);
  v26 = v17;
  if ( v17 )
  {
    v21 = (unsigned __int8)v13;
    v22 = v10;
    v23 = v12;
    sub_1623A60((__int64)&v26, v17, 2);
    v15 = v21;
    v10 = v22;
    v12 = v23;
  }
  v18 = *(_QWORD *)(v8 + 88);
  v19 = *(__int64 **)a1;
  v24 = v10;
  v27 = *(_DWORD *)(a2 + 64);
  *(_QWORD *)&v20 = sub_1D360F0(v19, v18, (__int64)&v26, v15, v12, 0, a3, a4, a5);
  result = sub_1D309E0(*(__int64 **)a1, 157, (__int64)&v26, v11, v24, 0, a3, a4, *(double *)a5.m128i_i64, v20);
  if ( v26 )
  {
    v25 = result;
    sub_161E7C0((__int64)&v26, v26);
    return v25;
  }
  return result;
}
