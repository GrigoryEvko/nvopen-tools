// Function: sub_200D960
// Address: 0x200d960
//
unsigned __int64 __fastcall sub_200D960(
        __int64 *a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __m128i a6,
        double a7,
        __m128i a8)
{
  __int64 v11; // rsi
  unsigned int v12; // ecx
  __int64 v13; // rdx
  __int64 v14; // rsi
  unsigned __int8 *v15; // rax
  unsigned int v16; // ebx
  __int128 v17; // rax
  int v18; // edx
  __int64 *v19; // r15
  __int128 v20; // rax
  __int64 *v21; // rax
  __int64 v22; // rsi
  unsigned int v23; // edx
  unsigned __int64 result; // rax
  __int64 *v25; // [rsp+0h] [rbp-A0h]
  unsigned int v26; // [rsp+8h] [rbp-98h]
  const void **v27; // [rsp+8h] [rbp-98h]
  __int64 v30; // [rsp+40h] [rbp-60h] BYREF
  int v31; // [rsp+48h] [rbp-58h]
  _BYTE v32[16]; // [rsp+50h] [rbp-50h] BYREF
  const void **v33; // [rsp+60h] [rbp-40h]

  v11 = *(_QWORD *)(a2 + 72);
  v12 = a3;
  v30 = v11;
  if ( v11 )
  {
    v26 = a3;
    sub_1623A60((__int64)&v30, v11, 2);
    v12 = v26;
  }
  v13 = a1[1];
  v14 = *a1;
  v31 = *(_DWORD *)(a2 + 64);
  v15 = (unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL * v12);
  sub_1F40D10((__int64)v32, v14, *(_QWORD *)(v13 + 48), *v15, *((_QWORD *)v15 + 1));
  v16 = v32[8];
  v27 = v33;
  v25 = (__int64 *)a1[1];
  *(_QWORD *)&v17 = sub_1D38E70((__int64)v25, 0, (__int64)&v30, 0, a6, a7, a8);
  *(_QWORD *)a4 = sub_1D332F0(v25, 49, (__int64)&v30, v16, v27, 0, *(double *)a6.m128i_i64, a7, a8, a2, a3, v17);
  *(_DWORD *)(a4 + 8) = v18;
  v19 = (__int64 *)a1[1];
  *(_QWORD *)&v20 = sub_1D38E70((__int64)v19, 1, (__int64)&v30, 0, a6, a7, a8);
  v21 = sub_1D332F0(v19, 49, (__int64)&v30, v16, v27, 0, *(double *)a6.m128i_i64, a7, a8, a2, a3, v20);
  v22 = v30;
  *(_QWORD *)a5 = v21;
  result = v23;
  *(_DWORD *)(a5 + 8) = v23;
  if ( v22 )
    return sub_161E7C0((__int64)&v30, v22);
  return result;
}
