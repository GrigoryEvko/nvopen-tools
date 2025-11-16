// Function: sub_20756A0
// Address: 0x20756a0
//
void __fastcall sub_20756A0(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  int v6; // edx
  __int64 v7; // rax
  __int64 v8; // rsi
  unsigned __int8 v9; // r13
  __int128 v10; // rax
  _QWORD *v11; // r15
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 *v14; // rax
  unsigned int v15; // edx
  __int64 *v16; // r13
  int v17; // edx
  __int64 *v18; // rax
  __int64 v19; // r12
  unsigned int v20; // [rsp+0h] [rbp-C0h]
  int v21; // [rsp+4h] [rbp-BCh]
  __int64 *v22; // [rsp+8h] [rbp-B8h]
  __int128 v23; // [rsp+10h] [rbp-B0h]
  __int64 *v24; // [rsp+20h] [rbp-A0h]
  __int64 v25; // [rsp+28h] [rbp-98h]
  __int128 v26; // [rsp+30h] [rbp-90h]
  __int64 v27; // [rsp+78h] [rbp-48h] BYREF
  __int64 v28; // [rsp+80h] [rbp-40h] BYREF
  int v29; // [rsp+88h] [rbp-38h]

  v6 = *(_DWORD *)(a1 + 536);
  v7 = *(_QWORD *)a1;
  v28 = 0;
  v29 = v6;
  if ( v7 )
  {
    if ( &v28 != (__int64 *)(v7 + 48) )
    {
      v8 = *(_QWORD *)(v7 + 48);
      v28 = v8;
      if ( v8 )
        sub_1623A60((__int64)&v28, v8, 2);
    }
  }
  v9 = *(_BYTE *)(a2 + 56);
  v20 = *(unsigned __int16 *)(a2 + 18);
  v21 = dword_4307AC0[(v20 >> 5) & 0x3FF];
  *(_QWORD *)&v10 = sub_2051C20((__int64 *)a1, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
  v11 = *(_QWORD **)(a1 + 552);
  v23 = v10;
  v22 = *(__int64 **)(a2 - 48);
  v24 = sub_20685E0(a1, *(__int64 **)(a2 - 24), a3, a4, a5);
  v25 = v12;
  *(_QWORD *)&v26 = sub_20685E0(a1, *(__int64 **)(a2 - 48), a3, a4, a5);
  *((_QWORD *)&v26 + 1) = v13;
  v14 = sub_20685E0(a1, *(__int64 **)(a2 - 24), a3, a4, a5);
  v16 = sub_1D2B9C0(
          v11,
          v21,
          (__int64)&v28,
          *(unsigned __int8 *)(v14[5] + 16LL * v15),
          0,
          v22,
          v23,
          v26,
          (__int64)v24,
          v25,
          0,
          (v20 >> 2) & 7,
          v9);
  LODWORD(v11) = v17;
  v27 = a2;
  v18 = sub_205F5C0(a1 + 8, &v27);
  v18[1] = (__int64)v16;
  *((_DWORD *)v18 + 4) = (_DWORD)v11;
  v19 = *(_QWORD *)(a1 + 552);
  if ( v16 )
  {
    nullsub_686();
    *(_QWORD *)(v19 + 176) = v16;
    *(_DWORD *)(v19 + 184) = 1;
    sub_1D23870();
  }
  else
  {
    *(_QWORD *)(v19 + 176) = 0;
    *(_DWORD *)(v19 + 184) = 1;
  }
  if ( v28 )
    sub_161E7C0((__int64)&v28, v28);
}
