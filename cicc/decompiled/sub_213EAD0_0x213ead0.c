// Function: sub_213EAD0
// Address: 0x213ead0
//
__int64 __fastcall sub_213EAD0(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 v6; // rsi
  unsigned int v7; // r15d
  __int64 v8; // rsi
  unsigned __int8 *v9; // rax
  __int16 v10; // ax
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // r14
  __int128 v16; // [rsp-10h] [rbp-C0h]
  const void **v17; // [rsp+18h] [rbp-98h]
  __int64 v18; // [rsp+50h] [rbp-60h] BYREF
  int v19; // [rsp+58h] [rbp-58h]
  _BYTE v20[16]; // [rsp+60h] [rbp-50h] BYREF
  const void **v21; // [rsp+70h] [rbp-40h]

  sub_1F40D10(
    (__int64)v20,
    *(_QWORD *)a1,
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v6 = *(_QWORD *)(a2 + 72);
  v7 = v20[8];
  v17 = v21;
  v18 = v6;
  if ( v6 )
    sub_1623A60((__int64)&v18, v6, 2);
  v8 = *(_QWORD *)a1;
  v19 = *(_DWORD *)(a2 + 64);
  v9 = (unsigned __int8 *)(*(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL)
                         + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL));
  sub_1F40D10((__int64)v20, v8, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), *v9, *((_QWORD *)v9 + 1));
  if ( v20[0] == 1 )
  {
    v10 = *(_WORD *)(a2 + 24);
    if ( v10 == 150 )
    {
      v12 = (__int64)sub_2139100(
                       a1,
                       **(_QWORD **)(a2 + 32),
                       *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
                       *(double *)a3.m128i_i64,
                       a4,
                       a5);
      v11 = (unsigned int)v11;
    }
    else
    {
      if ( v10 == 151 )
        v12 = (__int64)sub_2139210(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), a3, a4, a5);
      else
        v12 = sub_2138AD0(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
      v11 = (unsigned int)v11;
    }
    *((_QWORD *)&v16 + 1) = v11;
    *(_QWORD *)&v16 = v12;
    v13 = sub_1D309E0(
            *(__int64 **)(a1 + 8),
            *(unsigned __int16 *)(a2 + 24),
            (__int64)&v18,
            v7,
            v17,
            0,
            *(double *)a3.m128i_i64,
            a4,
            *(double *)a5.m128i_i64,
            v16);
  }
  else
  {
    v13 = sub_1D309E0(
            *(__int64 **)(a1 + 8),
            *(unsigned __int16 *)(a2 + 24),
            (__int64)&v18,
            v7,
            v17,
            0,
            *(double *)a3.m128i_i64,
            a4,
            *(double *)a5.m128i_i64,
            *(_OWORD *)*(_QWORD *)(a2 + 32));
  }
  v14 = v13;
  if ( v18 )
    sub_161E7C0((__int64)&v18, v18);
  return v14;
}
