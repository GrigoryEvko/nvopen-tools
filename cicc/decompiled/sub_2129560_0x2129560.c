// Function: sub_2129560
// Address: 0x2129560
//
__int64 *__fastcall sub_2129560(_QWORD *a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 v5; // r10
  const __m128i *v7; // rax
  __int64 v8; // rsi
  __m128i v9; // xmm0
  __int64 v10; // r8
  unsigned __int32 v11; // ecx
  __int64 v12; // r14
  __int64 v13; // r15
  __int64 v14; // r12
  unsigned __int64 v15; // r13
  _DWORD *v16; // r9
  __int64 v17; // rsi
  unsigned __int8 *v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 *v21; // r10
  const void **v22; // rdx
  const void **v23; // r9
  __int64 v24; // r8
  _DWORD *v25; // rdx
  bool v26; // cl
  int v27; // eax
  __int64 *v28; // rdi
  __int64 *v29; // r12
  bool v31; // al
  __int64 v32; // rdx
  __int128 v33; // [rsp-10h] [rbp-D0h]
  __int64 *v34; // [rsp+0h] [rbp-C0h]
  __int64 v35; // [rsp+0h] [rbp-C0h]
  __int64 v36; // [rsp+8h] [rbp-B8h]
  const void **v37; // [rsp+8h] [rbp-B8h]
  __int64 v38; // [rsp+10h] [rbp-B0h]
  __int64 *v39; // [rsp+10h] [rbp-B0h]
  __int64 v40; // [rsp+18h] [rbp-A8h]
  _DWORD *v41; // [rsp+18h] [rbp-A8h]
  _DWORD *v42; // [rsp+18h] [rbp-A8h]
  __int64 v43; // [rsp+20h] [rbp-A0h]
  __int64 (__fastcall *v44)(_DWORD *, __int64, __int64, __int64, __int64); // [rsp+20h] [rbp-A0h]
  __int32 v45; // [rsp+28h] [rbp-98h]
  __int64 v46; // [rsp+28h] [rbp-98h]
  bool v47; // [rsp+28h] [rbp-98h]
  __int64 *v48; // [rsp+28h] [rbp-98h]
  __int64 v49; // [rsp+70h] [rbp-50h] BYREF
  int v50; // [rsp+78h] [rbp-48h]
  _QWORD v51[8]; // [rsp+80h] [rbp-40h] BYREF

  v5 = a2;
  v7 = *(const __m128i **)(a2 + 32);
  v8 = *(_QWORD *)(a2 + 72);
  v9 = _mm_loadu_si128(v7);
  v10 = v7->m128i_i64[0];
  v49 = v8;
  v11 = v7->m128i_u32[2];
  v12 = v7[2].m128i_i64[1];
  v13 = v7[3].m128i_i64[0];
  v14 = v7[5].m128i_i64[0];
  v15 = v7[5].m128i_u64[1];
  if ( v8 )
  {
    v40 = v5;
    v43 = v10;
    v45 = v7->m128i_i32[2];
    sub_1623A60((__int64)&v49, v8, 2);
    v5 = v40;
    v10 = v43;
    v11 = v45;
  }
  v16 = (_DWORD *)*a1;
  v34 = (__int64 *)v5;
  v17 = a1[1];
  v50 = *(_DWORD *)(v5 + 64);
  v18 = (unsigned __int8 *)(*(_QWORD *)(v10 + 40) + 16LL * v11);
  v41 = v16;
  v46 = *(_QWORD *)(v17 + 48);
  v36 = *((_QWORD *)v18 + 1);
  v38 = *v18;
  v44 = *(__int64 (__fastcall **)(_DWORD *, __int64, __int64, __int64, __int64))(*(_QWORD *)v16 + 264LL);
  v19 = sub_1E0A0C0(*(_QWORD *)(v17 + 32));
  v20 = v44(v41, v19, v46, v38, v36);
  v21 = v34;
  v23 = v22;
  v24 = v20;
  v25 = (_DWORD *)*a1;
  v51[0] = v20;
  v51[1] = v23;
  if ( (_BYTE)v20 )
  {
    if ( (unsigned __int8)(v20 - 14) > 0x5Fu )
    {
      v26 = (unsigned __int8)(v20 - 86) <= 0x17u || (unsigned __int8)(v20 - 8) <= 5u;
      goto LABEL_6;
    }
LABEL_15:
    v27 = v25[17];
    v28 = (__int64 *)a1[1];
    if ( v27 != 1 )
      goto LABEL_9;
LABEL_16:
    v48 = v21;
    v14 = sub_1D323C0(v28, v14, v15, (__int64)&v49, v24, v23, *(double *)v9.m128i_i64, a4, a5);
    v32 = (unsigned int)v32;
    goto LABEL_17;
  }
  v35 = v20;
  v37 = v23;
  v39 = v21;
  v42 = v25;
  v47 = sub_1F58CD0((__int64)v51);
  v31 = sub_1F58D20((__int64)v51);
  v26 = v47;
  v25 = v42;
  v21 = v39;
  v23 = v37;
  v24 = v35;
  if ( v31 )
    goto LABEL_15;
LABEL_6:
  if ( v26 )
    v27 = v25[16];
  else
    v27 = v25[15];
  v28 = (__int64 *)a1[1];
  if ( v27 == 1 )
    goto LABEL_16;
LABEL_9:
  if ( v27 == 2 )
  {
    v48 = v21;
    v14 = sub_1D322C0(v28, v14, v15, (__int64)&v49, v24, v23, *(double *)v9.m128i_i64, a4, a5);
    v32 = (unsigned int)v32;
  }
  else
  {
    if ( v27 )
      goto LABEL_11;
    v48 = v21;
    v14 = sub_1D321C0(v28, v14, v15, (__int64)&v49, v24, v23, *(double *)v9.m128i_i64, a4, a5);
    v32 = (unsigned int)v32;
  }
LABEL_17:
  v28 = (__int64 *)a1[1];
  v21 = v48;
  v15 = v32 | v15 & 0xFFFFFFFF00000000LL;
LABEL_11:
  *((_QWORD *)&v33 + 1) = v15;
  *(_QWORD *)&v33 = v14;
  v29 = sub_1D2E2F0(v28, v21, v9.m128i_i64[0], v9.m128i_i64[1], v12, v13, v33);
  if ( v49 )
    sub_161E7C0((__int64)&v49, v49);
  return v29;
}
