// Function: sub_207E6C0
// Address: 0x207e6c0
//
void __fastcall sub_207E6C0(
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
  __int64 v10; // r14
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  _QWORD *v16; // r12
  __int64 v17; // rdx
  __int64 v18; // r13
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // r9
  __int64 *v23; // r10
  __int64 v24; // r11
  __int64 v25; // rax
  __int64 v26; // rsi
  int v27; // edx
  __int64 *v28; // r12
  int v29; // r13d
  __int128 v30; // [rsp-50h] [rbp-F0h]
  __int128 v31; // [rsp-20h] [rbp-C0h]
  __int64 *v32; // [rsp+0h] [rbp-A0h]
  __int64 v33; // [rsp+8h] [rbp-98h]
  __int128 v34; // [rsp+10h] [rbp-90h]
  __int128 v35; // [rsp+20h] [rbp-80h]
  __int128 v36; // [rsp+30h] [rbp-70h]
  __int64 v37; // [rsp+60h] [rbp-40h] BYREF
  int v38; // [rsp+68h] [rbp-38h]

  v10 = *(_QWORD *)(a1 + 552);
  v11 = 1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  *(_QWORD *)&v34 = sub_1D2AD90((_QWORD *)v10, *(_QWORD *)(a2 + 24 * v11), v11, a7, 1, a9);
  *((_QWORD *)&v34 + 1) = v12;
  v16 = sub_1D2AD90(
          *(_QWORD **)(a1 + 552),
          *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)),
          4LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF),
          v13,
          v14,
          v15);
  v18 = v17;
  *(_QWORD *)&v35 = sub_20685E0(a1, *(__int64 **)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))), a3, a4, a5);
  *((_QWORD *)&v35 + 1) = v19;
  *(_QWORD *)&v36 = sub_20685E0(a1, *(__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)), a3, a4, a5);
  *((_QWORD *)&v36 + 1) = v20;
  v37 = 0;
  v23 = sub_2051C20((__int64 *)a1, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
  v24 = v21;
  v25 = *(_QWORD *)a1;
  v38 = *(_DWORD *)(a1 + 536);
  if ( v25 )
  {
    if ( &v37 != (__int64 *)(v25 + 48) )
    {
      v26 = *(_QWORD *)(v25 + 48);
      v37 = v26;
      if ( v26 )
      {
        v32 = v23;
        v33 = v21;
        sub_1623A60((__int64)&v37, v26, 2);
        v23 = v32;
        v24 = v33;
      }
    }
  }
  *((_QWORD *)&v31 + 1) = v18;
  *(_QWORD *)&v31 = v16;
  *((_QWORD *)&v30 + 1) = v24;
  *(_QWORD *)&v30 = v23;
  v28 = sub_1D36A20((__int64 *)v10, 205, (__int64)&v37, 1, 0, v22, v30, v36, v35, v31, v34);
  v29 = v27;
  if ( v28 )
  {
    nullsub_686();
    *(_QWORD *)(v10 + 176) = v28;
    *(_DWORD *)(v10 + 184) = v29;
    sub_1D23870();
  }
  else
  {
    *(_QWORD *)(v10 + 176) = 0;
    *(_DWORD *)(v10 + 184) = v27;
  }
  if ( v37 )
    sub_161E7C0((__int64)&v37, v37);
}
