// Function: sub_207DD20
// Address: 0x207dd20
//
void __fastcall sub_207DD20(
        __int64 a1,
        __int64 a2,
        __m128i a3,
        __m128i a4,
        __m128i a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v9; // rbx
  _QWORD *v10; // r14
  __int64 v11; // rdx
  __int64 v12; // r15
  __int64 v13; // rdx
  __int16 *v14; // rdx
  __int64 *v15; // r8
  __int16 *v16; // r9
  __int64 v17; // rax
  __int64 v18; // rsi
  int v19; // edx
  __int64 *v20; // r12
  int v21; // r13d
  __int64 *v22; // [rsp+0h] [rbp-90h]
  __int16 *v23; // [rsp+8h] [rbp-88h]
  __int128 v24; // [rsp+20h] [rbp-70h]
  __int64 v25; // [rsp+50h] [rbp-40h] BYREF
  int v26; // [rsp+58h] [rbp-38h]

  v9 = *(_QWORD *)(a1 + 552);
  v10 = sub_1D2AD90(
          (_QWORD *)v9,
          *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)),
          4LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF),
          a7,
          a8,
          a9);
  v12 = v11;
  *(_QWORD *)&v24 = sub_20685E0(a1, *(__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)), a3, a4, a5);
  *((_QWORD *)&v24 + 1) = v13;
  v25 = 0;
  v15 = sub_2051C20((__int64 *)a1, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
  v16 = v14;
  v17 = *(_QWORD *)a1;
  v26 = *(_DWORD *)(a1 + 536);
  if ( v17 )
  {
    if ( &v25 != (__int64 *)(v17 + 48) )
    {
      v18 = *(_QWORD *)(v17 + 48);
      v25 = v18;
      if ( v18 )
      {
        v22 = v15;
        v23 = v14;
        sub_1623A60((__int64)&v25, v18, 2);
        v15 = v22;
        v16 = v23;
      }
    }
  }
  v20 = sub_1D3A900(
          (__int64 *)v9,
          0xCFu,
          (__int64)&v25,
          1u,
          0,
          0,
          (__m128)a3,
          *(double *)a4.m128i_i64,
          a5,
          (unsigned __int64)v15,
          v16,
          v24,
          (__int64)v10,
          v12);
  v21 = v19;
  if ( v20 )
  {
    nullsub_686();
    *(_QWORD *)(v9 + 176) = v20;
    *(_DWORD *)(v9 + 184) = v21;
    sub_1D23870();
  }
  else
  {
    *(_QWORD *)(v9 + 176) = 0;
    *(_DWORD *)(v9 + 184) = v19;
  }
  if ( v25 )
    sub_161E7C0((__int64)&v25, v25);
}
