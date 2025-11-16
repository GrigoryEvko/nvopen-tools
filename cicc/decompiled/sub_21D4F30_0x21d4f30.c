// Function: sub_21D4F30
// Address: 0x21d4f30
//
__int64 __fastcall sub_21D4F30(__m128i a1, double a2, __m128i a3, __int64 a4, __int64 a5, __int64 a6, __int64 *a7)
{
  _QWORD *v9; // rcx
  int v10; // eax
  __int64 result; // rax
  int v12; // eax
  __int64 v13; // r13
  void *v14; // r14
  __int64 *v15; // rsi
  __int64 *v16; // rsi
  __int64 v17; // rsi
  unsigned int v18; // ecx
  unsigned __int64 v19; // rax
  unsigned int v20; // edx
  __int64 v21; // rax
  __int64 v22; // r14
  __int64 v23; // rdx
  __int64 v24; // r15
  __int64 v25; // rsi
  __int128 v26; // [rsp-10h] [rbp-C0h]
  __int64 v27; // [rsp+8h] [rbp-A8h]
  __int64 v28; // [rsp+8h] [rbp-A8h]
  __int64 v29; // [rsp+8h] [rbp-A8h]
  const void *v30; // [rsp+10h] [rbp-A0h] BYREF
  unsigned int v31; // [rsp+18h] [rbp-98h]
  const void *v32; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v33; // [rsp+28h] [rbp-88h]
  unsigned __int64 v34; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v35; // [rsp+38h] [rbp-78h]
  unsigned __int64 v36; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v37; // [rsp+48h] [rbp-68h]
  __int64 v38; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v39; // [rsp+58h] [rbp-58h]
  __int64 v40; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v41; // [rsp+68h] [rbp-48h]
  __int64 v42; // [rsp+70h] [rbp-40h] BYREF
  int v43; // [rsp+78h] [rbp-38h]

  if ( **(_BYTE **)(a5 + 40) != 86 )
    return a5;
  v9 = *(_QWORD **)(a5 + 32);
  v10 = *(unsigned __int16 *)(*v9 + 24LL);
  if ( v10 != 33 && v10 != 11 )
    return a5;
  v12 = *(unsigned __int16 *)(v9[5] + 24LL);
  if ( v12 != 11 && v12 != 33 )
    return a5;
  v13 = *(_QWORD *)(*v9 + 88LL);
  v14 = sub_16982C0();
  v15 = (__int64 *)(v13 + 32);
  if ( *(void **)(v13 + 32) == v14 )
    sub_169D930((__int64)&v30, (__int64)v15);
  else
    sub_169D7E0((__int64)&v30, v15);
  v16 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a5 + 32) + 40LL) + 88LL) + 32LL);
  if ( (void *)*v16 == v14 )
    sub_169D930((__int64)&v32, (__int64)v16);
  else
    sub_169D7E0((__int64)&v32, v16);
  v17 = *(_QWORD *)(a5 + 72);
  v42 = v17;
  if ( v17 )
    sub_1623A60((__int64)&v42, v17, 2);
  v43 = *(_DWORD *)(a5 + 64);
  sub_16A5C50((__int64)&v38, &v30, 0x20u);
  sub_16A5C50((__int64)&v34, &v32, 0x20u);
  v18 = v35;
  v37 = v35;
  if ( v35 <= 0x40 )
  {
    v36 = v34;
LABEL_16:
    v19 = 0;
    if ( v18 != 16 )
      v19 = (v36 << 16) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v18);
    v36 = v19;
    goto LABEL_19;
  }
  sub_16A4FD0((__int64)&v36, (const void **)&v34);
  v18 = v37;
  if ( v37 <= 0x40 )
    goto LABEL_16;
  sub_16A7DC0((__int64 *)&v36, 0x10u);
LABEL_19:
  v20 = v39;
  if ( v39 > 0x40 )
  {
    sub_16A89F0(&v38, (__int64 *)&v36);
    v20 = v39;
    v21 = v38;
  }
  else
  {
    v21 = v36 | v38;
    v38 |= v36;
  }
  v41 = v20;
  v40 = v21;
  v39 = 0;
  v22 = sub_1D38970((__int64)a7, (__int64)&v40, (__int64)&v42, 5u, 0, 0, a1, a2, a3, 0);
  v24 = v23;
  if ( v41 > 0x40 && v40 )
    j_j___libc_free_0_0(v40);
  if ( v37 > 0x40 && v36 )
    j_j___libc_free_0_0(v36);
  if ( v35 > 0x40 && v34 )
    j_j___libc_free_0_0(v34);
  if ( v39 > 0x40 && v38 )
    j_j___libc_free_0_0(v38);
  if ( v42 )
    sub_161E7C0((__int64)&v42, v42);
  v25 = *(_QWORD *)(a5 + 72);
  v42 = v25;
  if ( v25 )
    sub_1623A60((__int64)&v42, v25, 2);
  *((_QWORD *)&v26 + 1) = v24;
  *(_QWORD *)&v26 = v22;
  v43 = *(_DWORD *)(a5 + 64);
  result = sub_1D309E0(a7, 158, (__int64)&v42, 86, 0, 0, *(double *)a1.m128i_i64, a2, *(double *)a3.m128i_i64, v26);
  if ( v42 )
  {
    v27 = result;
    sub_161E7C0((__int64)&v42, v42);
    result = v27;
  }
  if ( v33 > 0x40 && v32 )
  {
    v28 = result;
    j_j___libc_free_0_0(v32);
    result = v28;
  }
  if ( v31 > 0x40 )
  {
    if ( v30 )
    {
      v29 = result;
      j_j___libc_free_0_0(v30);
      return v29;
    }
  }
  return result;
}
