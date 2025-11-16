// Function: sub_2128400
// Address: 0x2128400
//
__int64 __fastcall sub_2128400(__int64 a1, __int64 a2, __m128 a3, double a4, __m128i a5)
{
  __int64 v7; // rax
  unsigned __int8 v8; // r13
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 (__fastcall *v11)(__int64, __int64, __int64, __int64, __int64); // r13
  __int64 v12; // rax
  unsigned int v13; // eax
  unsigned int v14; // r13d
  __int64 v15; // rdx
  const void **v16; // r8
  __int64 v17; // rsi
  unsigned int v18; // esi
  __int64 *v19; // rdi
  __int64 *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r12
  __int64 v24; // r14
  __int64 v25; // rax
  __int64 (__fastcall *v26)(__int64, __int64, __int64, __int64, const void **); // r13
  __int64 v27; // rax
  unsigned int v28; // eax
  const void **v29; // rdx
  __int64 v30; // [rsp+8h] [rbp-78h]
  const void **v31; // [rsp+8h] [rbp-78h]
  __int64 v32; // [rsp+10h] [rbp-70h]
  const void **v33; // [rsp+10h] [rbp-70h]
  const void **v34; // [rsp+10h] [rbp-70h]
  __int64 v35; // [rsp+10h] [rbp-70h]
  unsigned __int8 v36; // [rsp+1Eh] [rbp-62h]
  unsigned __int8 v37; // [rsp+1Fh] [rbp-61h]
  const void **v38; // [rsp+20h] [rbp-60h]
  __int64 v39; // [rsp+28h] [rbp-58h]
  unsigned __int8 v40; // [rsp+28h] [rbp-58h]
  __int64 v41; // [rsp+28h] [rbp-58h]
  __int64 v42; // [rsp+30h] [rbp-50h] BYREF
  int v43; // [rsp+38h] [rbp-48h]
  const void **v44; // [rsp+40h] [rbp-40h]

  v7 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL);
  v8 = *(_BYTE *)v7;
  v9 = *(_QWORD *)(v7 + 8);
  v36 = *(_BYTE *)v7;
  sub_1F40D10(
    (__int64)&v42,
    *(_QWORD *)a1,
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v37 = v43;
  v30 = v8;
  v38 = v44;
  v10 = *(_QWORD *)(a1 + 8);
  v11 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(**(_QWORD **)a1 + 264LL);
  v32 = *(_QWORD *)a1;
  v39 = *(_QWORD *)(v10 + 48);
  v12 = sub_1E0A0C0(*(_QWORD *)(v10 + 32));
  v13 = v11(v32, v12, v39, v30, v9);
  v40 = v13;
  v14 = v13;
  v33 = (const void **)v15;
  sub_1F40D10((__int64)&v42, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), (unsigned __int8)v13, v15);
  v16 = v33;
  if ( (_BYTE)v42 == 1 )
  {
    sub_1F40D10((__int64)&v42, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), v36, v9);
    if ( (_BYTE)v42 == 1 )
    {
      sub_1F40D10((__int64)&v42, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), v36, v9);
      v24 = *(_QWORD *)a1;
      v25 = *(_QWORD *)(a1 + 8);
      v31 = v44;
      v26 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, const void **))(**(_QWORD **)a1 + 264LL);
      v35 = (unsigned __int8)v43;
      v41 = *(_QWORD *)(v25 + 48);
      v27 = sub_1E0A0C0(*(_QWORD *)(v25 + 32));
      v28 = v26(v24, v27, v41, v35, v31);
      v40 = v28;
      v14 = v28;
      v16 = v29;
    }
    else
    {
      v16 = v38;
      v40 = v37;
    }
  }
  v17 = *(_QWORD *)(a2 + 72);
  v42 = v17;
  if ( v17 )
  {
    v34 = v16;
    sub_1623A60((__int64)&v42, v17, 2);
    v16 = v34;
  }
  LOBYTE(v14) = v40;
  v18 = *(unsigned __int16 *)(a2 + 24);
  v19 = *(__int64 **)(a1 + 8);
  v43 = *(_DWORD *)(a2 + 64);
  v20 = sub_1D3A900(
          v19,
          v18,
          (__int64)&v42,
          v14,
          v16,
          0,
          a3,
          a4,
          a5,
          **(_QWORD **)(a2 + 32),
          *(__int16 **)(*(_QWORD *)(a2 + 32) + 8LL),
          *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL),
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
  v22 = sub_1D322C0(
          *(__int64 **)(a1 + 8),
          (__int64)v20,
          v21,
          (__int64)&v42,
          v37,
          v38,
          *(double *)a3.m128_u64,
          a4,
          *(double *)a5.m128i_i64);
  if ( v42 )
    sub_161E7C0((__int64)&v42, v42);
  return v22;
}
