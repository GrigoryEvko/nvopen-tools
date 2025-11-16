// Function: sub_1F972B0
// Address: 0x1f972b0
//
__int64 *__fastcall sub_1F972B0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __m128i a6,
        double a7,
        __m128i a8)
{
  unsigned __int8 *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rdi
  unsigned int v15; // r14d
  int v16; // eax
  __int64 v17; // rax
  __int64 *v18; // r13
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // r15
  __int64 v22; // [rsp+0h] [rbp-60h]
  char v23; // [rsp+1Fh] [rbp-41h] BYREF
  __int64 v24; // [rsp+20h] [rbp-40h] BYREF
  int v25; // [rsp+28h] [rbp-38h]

  v12 = (unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3);
  v13 = *(_QWORD *)(a2 + 72);
  v14 = *((_QWORD *)v12 + 1);
  v15 = *v12;
  v24 = v13;
  if ( v13 )
  {
    v22 = a4;
    sub_1623A60((__int64)&v24, v13, 2);
    a4 = v22;
  }
  v16 = *(_DWORD *)(a2 + 64);
  v23 = 0;
  v25 = v16;
  v17 = sub_1F973B0(a1, a2, a3, a4, a5, &v23);
  v18 = (__int64 *)v17;
  v20 = v19;
  if ( v17 )
  {
    sub_1F81BC0((__int64)a1, v17);
    if ( v23 )
      sub_1F97190(a1, a2, (__int64)v18, *(double *)a6.m128i_i64, a7, *(double *)a8.m128i_i64);
    v18 = sub_1D3BC50((__int64 *)*a1, (__int64)v18, v20, (__int64)&v24, v15, v14, a6, a7, a8);
  }
  if ( v24 )
    sub_161E7C0((__int64)&v24, v24);
  return v18;
}
