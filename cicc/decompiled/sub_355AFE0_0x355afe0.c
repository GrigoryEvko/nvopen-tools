// Function: sub_355AFE0
// Address: 0x355afe0
//
__int64 *__fastcall sub_355AFE0(__int64 *a1, _QWORD *a2, __int64 *a3)
{
  __int64 v6; // rdx
  __int64 v7; // r15
  __int64 v8; // r10
  __int64 v9; // r9
  __int64 v10; // rdi
  __int64 v11; // r14
  __int64 v12; // rsi
  unsigned __int64 v13; // r11
  __int64 v14; // rcx
  unsigned __int64 v15; // r12
  __int64 v16; // rdx
  __int64 *v17; // rax
  __int64 v18; // rax
  __int64 *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // [rsp+0h] [rbp-F0h]
  __int64 v24; // [rsp+8h] [rbp-E8h]
  __int64 v25; // [rsp+10h] [rbp-E0h]
  __int64 v26; // [rsp+18h] [rbp-D8h]
  unsigned __int64 v27; // [rsp+28h] [rbp-C8h]
  __int64 v28; // [rsp+30h] [rbp-C0h]
  __int64 v29; // [rsp+38h] [rbp-B8h]
  __int64 v30[4]; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v31; // [rsp+60h] [rbp-90h] BYREF
  unsigned __int64 v32; // [rsp+68h] [rbp-88h]
  __int64 v33; // [rsp+70h] [rbp-80h]
  __int64 v34; // [rsp+78h] [rbp-78h]
  __int64 v35; // [rsp+80h] [rbp-70h] BYREF
  unsigned __int64 v36; // [rsp+88h] [rbp-68h]
  __int64 v37; // [rsp+90h] [rbp-60h]
  __int64 v38; // [rsp+98h] [rbp-58h]
  __m128i v39; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v40; // [rsp+B0h] [rbp-40h]
  __int64 v41; // [rsp+B8h] [rbp-38h]

  v6 = *a3;
  v7 = a3[1];
  v8 = a3[3];
  v9 = v6 + 8;
  v28 = a3[2];
  if ( v28 == v6 + 8 )
  {
    v9 = *(_QWORD *)(v8 + 8);
    v24 = v8 + 8;
    v26 = v9;
    v25 = v9 + 512;
  }
  else
  {
    v24 = a3[3];
    v26 = a3[1];
    v25 = a3[2];
  }
  v10 = a2[4];
  v11 = a2[5];
  v12 = a2[6];
  v13 = a2[3];
  v29 = a2[2];
  v27 = a2[7];
  v14 = (a2[4] - v29) >> 3;
  v15 = v14 + ((((v8 - v11) >> 3) - 1) << 6) + ((v6 - v7) >> 3);
  v23 = a2[9];
  if ( (unsigned __int64)(v14 + ((__int64)(v12 - v27) >> 3) + ((((v23 - v11) >> 3) - 1) << 6)) >> 1 <= v15 )
  {
    if ( v9 != v12 )
    {
      v18 = a2[8];
      v39.m128i_i64[0] = v6;
      v35 = v12;
      v37 = v18;
      v40 = v28;
      v38 = v23;
      v36 = v27;
      v32 = v26;
      v39.m128i_i64[1] = v7;
      v33 = v25;
      v41 = v8;
      v34 = v24;
      v31 = v9;
      sub_355AE80(v30, (__int64)&v31, &v35, (__int64)&v39);
      v12 = a2[6];
      v27 = a2[7];
    }
    if ( v27 == v12 )
    {
      j_j___libc_free_0(v27);
      v13 = a2[3];
      v10 = a2[4];
      v20 = (__int64 *)(a2[9] - 8LL);
      a2[9] = v20;
      v21 = *v20;
      v22 = *v20 + 512;
      a2[7] = v21;
      a2[8] = v22;
      v16 = a2[2];
      a2[6] = v21 + 504;
      v17 = (__int64 *)a2[5];
    }
    else
    {
      v16 = a2[2];
      v13 = a2[3];
      v10 = a2[4];
      v17 = (__int64 *)a2[5];
      a2[6] = v12 - 8;
    }
  }
  else
  {
    if ( v29 != v6 )
    {
      v35 = v6;
      v33 = v10;
      v39.m128i_i64[1] = v26;
      v32 = v13;
      v40 = v25;
      v39.m128i_i64[0] = v9;
      v41 = v24;
      v36 = v7;
      v37 = v28;
      v38 = v8;
      v31 = v29;
      v34 = v11;
      sub_355ABF0(v30, &v31, (__int64)&v35, &v39);
      v6 = a2[2];
      v10 = a2[4];
      v13 = a2[3];
    }
    if ( v6 == v10 - 8 )
    {
      j_j___libc_free_0(v13);
      v17 = (__int64 *)(a2[5] + 8LL);
      a2[5] = v17;
      v16 = *v17;
      v10 = *v17 + 512;
      a2[3] = *v17;
      v13 = v16;
      a2[4] = v10;
    }
    else
    {
      v16 = v6 + 8;
      v17 = (__int64 *)a2[5];
    }
    a2[2] = v16;
  }
  a1[2] = v10;
  a1[3] = (__int64)v17;
  *a1 = v16;
  a1[1] = v13;
  sub_353DF70(a1, v15);
  return a1;
}
