// Function: sub_37A8BC0
// Address: 0x37a8bc0
//
__m128i *__fastcall sub_37A8BC0(__int64 a1, __int64 a2, int a3)
{
  int v3; // r8d
  _QWORD *v5; // rax
  __int64 v6; // rsi
  unsigned __int64 v7; // r12
  __int64 v8; // r13
  unsigned __int64 v9; // r14
  __int64 v10; // r9
  __int64 v11; // r12
  unsigned int v12; // edx
  unsigned __int64 v13; // r13
  __int64 v14; // r14
  unsigned int v15; // edx
  unsigned __int64 v16; // r9
  __int64 v17; // rcx
  __m128i *v18; // r12
  unsigned int v20; // edx
  __int64 v21; // rax
  unsigned int v22; // edx
  __int128 v23; // [rsp-50h] [rbp-F0h]
  __int64 v24; // [rsp+8h] [rbp-98h]
  __int64 v26; // [rsp+10h] [rbp-90h]
  unsigned __int64 v27; // [rsp+10h] [rbp-90h]
  __int64 v28; // [rsp+60h] [rbp-40h] BYREF
  int v29; // [rsp+68h] [rbp-38h]

  v3 = a3;
  v5 = *(_QWORD **)(a2 + 40);
  v6 = *(_QWORD *)(a2 + 80);
  v7 = v5[20];
  v8 = v5[21];
  v28 = v6;
  v9 = v5[5];
  v10 = v5[6];
  if ( v6 )
  {
    v24 = v5[6];
    sub_B96E90((__int64)&v28, v6, 1);
    v10 = v24;
    v3 = a3;
  }
  v29 = *(_DWORD *)(a2 + 72);
  v26 = v10;
  if ( v3 == 1 )
  {
    v14 = sub_379AB60(a1, v9, v10);
    v27 = v20 | v26 & 0xFFFFFFFF00000000LL;
    v21 = sub_379AB60(a1, v7, v8);
    v16 = v27;
    v11 = v21;
    v13 = v22 | v8 & 0xFFFFFFFF00000000LL;
  }
  else
  {
    v11 = sub_379AB60(a1, v7, v8);
    v13 = v12 | v8 & 0xFFFFFFFF00000000LL;
    v14 = sub_379AB60(a1, v9, v26);
    v16 = v15 | v26 & 0xFFFFFFFF00000000LL;
  }
  v17 = *(_QWORD *)(a2 + 40);
  *((_QWORD *)&v23 + 1) = v13;
  *(_QWORD *)&v23 = v11;
  v18 = sub_33F51B0(
          *(__int64 **)(a1 + 8),
          *(_QWORD *)v17,
          *(_QWORD *)(v17 + 8),
          (__int64)&v28,
          v14,
          v16,
          *(_QWORD *)(v17 + 80),
          *(_QWORD *)(v17 + 88),
          *(_OWORD *)(v17 + 120),
          v23,
          *(_OWORD *)(v17 + 200),
          *(unsigned __int16 *)(a2 + 96),
          *(_QWORD *)(a2 + 104),
          *(const __m128i **)(a2 + 112),
          (*(_WORD *)(a2 + 32) >> 7) & 7,
          (*(_BYTE *)(a2 + 33) & 4) != 0,
          (*(_BYTE *)(a2 + 33) & 8) != 0);
  if ( v28 )
    sub_B91220((__int64)&v28, v28);
  return v18;
}
