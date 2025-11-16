// Function: sub_212FD50
// Address: 0x212fd50
//
unsigned __int64 __fastcall sub_212FD50(
        __int64 a1,
        __int64 a2,
        _DWORD *a3,
        _DWORD *a4,
        __m128i a5,
        double a6,
        __m128i a7)
{
  __int64 v10; // rsi
  unsigned __int8 *v11; // rax
  unsigned int v12; // r15d
  __int128 v13; // rax
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  int v16; // edx
  __int64 v17; // rax
  __int64 v18; // rsi
  int v19; // edx
  unsigned __int64 result; // rax
  unsigned __int64 v21; // [rsp-10h] [rbp-90h]
  __int64 *v22; // [rsp+0h] [rbp-80h]
  const void **v23; // [rsp+8h] [rbp-78h]
  __int128 v24; // [rsp+10h] [rbp-70h]
  __int64 v25; // [rsp+40h] [rbp-40h] BYREF
  int v26; // [rsp+48h] [rbp-38h]

  v10 = *(_QWORD *)(a2 + 72);
  v25 = v10;
  if ( v10 )
    sub_1623A60((__int64)&v25, v10, 2);
  v26 = *(_DWORD *)(a2 + 64);
  sub_20174B0(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), a3, a4);
  v11 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)a3 + 40LL) + 16LL * (unsigned int)a3[2]);
  v12 = *v11;
  v22 = *(__int64 **)(a1 + 8);
  v23 = (const void **)*((_QWORD *)v11 + 1);
  *(_QWORD *)&v13 = sub_1D309E0(
                      v22,
                      130,
                      (__int64)&v25,
                      *v11,
                      v23,
                      0,
                      *(double *)a5.m128i_i64,
                      a6,
                      *(double *)a7.m128i_i64,
                      *(_OWORD *)a4);
  v24 = v13;
  v14 = sub_1D309E0(
          *(__int64 **)(a1 + 8),
          130,
          (__int64)&v25,
          v12,
          v23,
          0,
          *(double *)a5.m128i_i64,
          a6,
          *(double *)a7.m128i_i64,
          *(_OWORD *)a3);
  *(_QWORD *)a3 = sub_1D332F0(v22, 52, (__int64)&v25, v12, v23, 0, *(double *)a5.m128i_i64, a6, a7, v14, v15, v24);
  a3[2] = v16;
  v17 = sub_1D38BB0(*(_QWORD *)(a1 + 8), 0, (__int64)&v25, v12, v23, 0, a5, a6, a7, 0);
  v18 = v25;
  *(_QWORD *)a4 = v17;
  a4[2] = v19;
  result = v21;
  if ( v18 )
    return sub_161E7C0((__int64)&v25, v18);
  return result;
}
