// Function: sub_202DEF0
// Address: 0x202def0
//
__int64 __fastcall sub_202DEF0(__int64 a1, __int64 a2, __m128 a3, double a4, __m128i a5)
{
  unsigned int v5; // r14d
  unsigned int v6; // r15d
  __int64 v8; // rsi
  __int64 v9; // rax
  char v10; // dl
  __int64 v11; // rax
  int v12; // ecx
  unsigned int v13; // eax
  const void **v14; // r8
  unsigned int v15; // ecx
  unsigned int v16; // eax
  const void **v17; // r8
  unsigned int v18; // edx
  __int64 *v19; // rax
  unsigned int v20; // edx
  __int64 *v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r14
  const void **v25; // rdx
  const void **v26; // rdx
  __int128 v27; // [rsp-10h] [rbp-F0h]
  _QWORD *v28; // [rsp+8h] [rbp-D8h]
  _QWORD *v29; // [rsp+10h] [rbp-D0h]
  unsigned int v30; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v31; // [rsp+18h] [rbp-C8h]
  const void **v32; // [rsp+20h] [rbp-C0h]
  const void **v33; // [rsp+20h] [rbp-C0h]
  unsigned int v34; // [rsp+28h] [rbp-B8h]
  const void **v35; // [rsp+28h] [rbp-B8h]
  __int64 *v36; // [rsp+40h] [rbp-A0h]
  unsigned __int64 v37; // [rsp+50h] [rbp-90h] BYREF
  __int16 *v38; // [rsp+58h] [rbp-88h]
  unsigned __int64 v39; // [rsp+60h] [rbp-80h] BYREF
  __int16 *v40; // [rsp+68h] [rbp-78h]
  __int128 v41; // [rsp+70h] [rbp-70h] BYREF
  __int128 v42; // [rsp+80h] [rbp-60h] BYREF
  __int64 v43; // [rsp+90h] [rbp-50h] BYREF
  int v44; // [rsp+98h] [rbp-48h]
  char v45[8]; // [rsp+A0h] [rbp-40h] BYREF
  __int64 v46; // [rsp+A8h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 72);
  v37 = 0;
  LODWORD(v38) = 0;
  v39 = 0;
  LODWORD(v40) = 0;
  *(_QWORD *)&v41 = 0;
  DWORD2(v41) = 0;
  *(_QWORD *)&v42 = 0;
  DWORD2(v42) = 0;
  v43 = v8;
  if ( v8 )
    sub_1623A60((__int64)&v43, v8, 2);
  v44 = *(_DWORD *)(a2 + 64);
  sub_2017DE0(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), &v37, &v39);
  sub_2017DE0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL), &v41, &v42);
  v9 = *(_QWORD *)(v37 + 40) + 16LL * (unsigned int)v38;
  v10 = *(_BYTE *)v9;
  v11 = *(_QWORD *)(v9 + 8);
  v45[0] = v10;
  v46 = v11;
  if ( v10 )
    v12 = word_4305480[(unsigned __int8)(v10 - 14)];
  else
    v12 = sub_1F58D30((__int64)v45);
  v34 = v12;
  v29 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 48LL);
  LOBYTE(v13) = sub_1D15020(2, v12);
  v14 = 0;
  v15 = v34;
  if ( !(_BYTE)v13 )
  {
    v13 = sub_1F593D0(v29, 2, 0, v34);
    v15 = v34;
    v5 = v13;
    v14 = v25;
  }
  LOBYTE(v5) = v13;
  v32 = v14;
  v30 = 2 * v15;
  v28 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 48LL);
  LOBYTE(v16) = sub_1D15020(2, 2 * v15);
  v35 = 0;
  v17 = v32;
  if ( !(_BYTE)v16 )
  {
    v16 = sub_1F593D0(v28, 2, 0, v30);
    v17 = v32;
    v35 = v26;
    v6 = v16;
  }
  LOBYTE(v6) = v16;
  v33 = v17;
  v36 = sub_1D3A900(
          *(__int64 **)(a1 + 8),
          0x89u,
          (__int64)&v43,
          v5,
          v17,
          0,
          a3,
          a4,
          a5,
          v37,
          v38,
          v41,
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL),
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
  v31 = v18;
  v19 = sub_1D3A900(
          *(__int64 **)(a1 + 8),
          0x89u,
          (__int64)&v43,
          v5,
          v33,
          0,
          a3,
          a4,
          a5,
          v39,
          v40,
          v42,
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL),
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
  *((_QWORD *)&v27 + 1) = v20;
  *(_QWORD *)&v27 = v19;
  v21 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          107,
          (__int64)&v43,
          v6,
          v35,
          0,
          *(double *)a3.m128_u64,
          a4,
          a5,
          (__int64)v36,
          v31,
          v27);
  v23 = sub_200E230(
          (_QWORD *)a1,
          (__int64)v21,
          v22,
          **(unsigned __int8 **)(a2 + 40),
          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
          *(double *)a3.m128_u64,
          a4,
          *(double *)a5.m128i_i64);
  if ( v43 )
    sub_161E7C0((__int64)&v43, v43);
  return v23;
}
