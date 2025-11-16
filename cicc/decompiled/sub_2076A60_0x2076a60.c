// Function: sub_2076A60
// Address: 0x2076a60
//
void __fastcall sub_2076A60(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  int v6; // edx
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 *v9; // r12
  __int64 v10; // rdx
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // rax
  unsigned int v15; // eax
  unsigned int v16; // ecx
  __int64 v17; // rdx
  unsigned int v18; // r14d
  int v19; // eax
  __int64 v20; // rdx
  __int128 v21; // rax
  __int64 *v22; // rax
  int v23; // edx
  __int64 v24; // r13
  __int64 *v25; // r12
  int v26; // r14d
  __int128 v27; // [rsp-50h] [rbp-F0h]
  _QWORD *v28; // [rsp+8h] [rbp-98h]
  __int64 *v29; // [rsp+10h] [rbp-90h]
  int v30; // [rsp+18h] [rbp-88h]
  unsigned __int8 v31; // [rsp+1Fh] [rbp-81h]
  __int64 v32; // [rsp+20h] [rbp-80h]
  __int64 *v33; // [rsp+20h] [rbp-80h]
  __int64 v34; // [rsp+28h] [rbp-78h]
  __int64 v35; // [rsp+50h] [rbp-50h] BYREF
  int v36; // [rsp+58h] [rbp-48h]
  unsigned int v37; // [rsp+60h] [rbp-40h] BYREF
  __int64 v38; // [rsp+68h] [rbp-38h]

  v6 = *(_DWORD *)(a1 + 536);
  v7 = *(_QWORD *)a1;
  v35 = 0;
  v36 = v6;
  if ( v7 )
  {
    if ( &v35 != (__int64 *)(v7 + 48) )
    {
      v8 = *(_QWORD *)(v7 + 48);
      v35 = v8;
      if ( v8 )
        sub_1623A60((__int64)&v35, v8, 2);
    }
  }
  v30 = (*(unsigned __int16 *)(a2 + 18) >> 7) & 7;
  v31 = *(_BYTE *)(a2 + 56);
  v9 = sub_2051C20((__int64 *)a1, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
  v11 = v10;
  v12 = *(_QWORD *)(a1 + 552);
  v13 = *(_QWORD *)(v12 + 16);
  v32 = **(_QWORD **)(a2 - 48);
  v14 = sub_1E0A0C0(*(_QWORD *)(v12 + 32));
  LOBYTE(v15) = sub_204D4D0(v13, v14, v32);
  v16 = *(unsigned __int16 *)(a2 + 18);
  v37 = v15;
  v38 = v17;
  v18 = 1 << (v16 >> 1) >> 1;
  if ( (_BYTE)v15 )
    v19 = sub_2045180(v15);
  else
    v19 = sub_1F58D40((__int64)&v37);
  if ( v18 < (unsigned int)(v19 + 7) >> 3 )
    sub_16BD130("Cannot generate unaligned atomic store", 1u);
  v28 = *(_QWORD **)(a1 + 552);
  v29 = *(__int64 **)(a2 - 24);
  v33 = sub_20685E0(a1, *(__int64 **)(a2 - 48), a3, a4, a5);
  v34 = v20;
  *(_QWORD *)&v21 = sub_20685E0(a1, *(__int64 **)(a2 - 24), a3, a4, a5);
  *((_QWORD *)&v27 + 1) = v11;
  *(_QWORD *)&v27 = v9;
  v22 = sub_1D2B9C0(v28, 220, (__int64)&v35, v37, v38, v29, v27, v21, (__int64)v33, v34, v18, v30, v31);
  v24 = *(_QWORD *)(a1 + 552);
  v25 = v22;
  v26 = v23;
  if ( v22 )
  {
    nullsub_686();
    *(_QWORD *)(v24 + 176) = v25;
    *(_DWORD *)(v24 + 184) = v26;
    sub_1D23870();
  }
  else
  {
    *(_QWORD *)(v24 + 176) = 0;
    *(_DWORD *)(v24 + 184) = v23;
  }
  if ( v35 )
    sub_161E7C0((__int64)&v35, v35);
}
