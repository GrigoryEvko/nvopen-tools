// Function: sub_27B9EC0
// Address: 0x27b9ec0
//
__int64 __fastcall sub_27B9EC0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r14d
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 v7; // rax
  __int64 v8; // r9
  __int64 v9; // rax
  const __m128i *v10; // rcx
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // r8
  __m128i *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rbx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // rbx
  unsigned __int64 v20; // r8
  unsigned __int64 v21; // rax
  __int64 v22; // r8
  bool v23; // bl
  unsigned int v24; // edx
  const void *v25; // rsi
  char *v26; // rbx
  unsigned __int64 v27; // [rsp+8h] [rbp-E8h]
  __int64 v28; // [rsp+18h] [rbp-D8h]
  const void **v29; // [rsp+18h] [rbp-D8h]
  __int64 v30; // [rsp+18h] [rbp-D8h]
  __int64 *v31; // [rsp+20h] [rbp-D0h]
  unsigned __int64 v32; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v33; // [rsp+30h] [rbp-C0h] BYREF
  unsigned int v34; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v35; // [rsp+40h] [rbp-B0h] BYREF
  unsigned __int32 v36; // [rsp+48h] [rbp-A8h]
  __int64 v37; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v38; // [rsp+58h] [rbp-98h]
  __int64 v39; // [rsp+60h] [rbp-90h]
  __int64 v40; // [rsp+68h] [rbp-88h]
  __m128i v41; // [rsp+70h] [rbp-80h] BYREF
  unsigned __int64 v42; // [rsp+80h] [rbp-70h]
  __int64 v43; // [rsp+88h] [rbp-68h]
  __int64 v44; // [rsp+90h] [rbp-60h]
  __int64 v45; // [rsp+98h] [rbp-58h]
  __int64 v46; // [rsp+A0h] [rbp-50h]
  __int64 v47; // [rsp+A8h] [rbp-48h]
  __int16 v48; // [rsp+B0h] [rbp-40h]

  if ( *(_BYTE *)a1 != 82 )
    return 0;
  v4 = *(_QWORD *)(a1 - 64);
  if ( *(_BYTE *)(*(_QWORD *)(v4 + 8) + 8LL) != 12 || (((*(_WORD *)(a1 + 2) & 0x3F) - 34) & 0xFFFD) != 0 )
    return 0;
  v5 = *(_QWORD *)(a1 - 32);
  if ( (*(_WORD *)(a1 + 2) & 0x3F) == 0x22 )
  {
    v4 = *(_QWORD *)(a1 - 32);
    v5 = *(_QWORD *)(a1 - 64);
  }
  v32 = sub_B43CC0(a1);
  v7 = sub_AD6530(*(_QWORD *)(v5 + 8), a2);
  v41 = (__m128i)v32;
  v48 = 257;
  v39 = v5;
  v37 = v4;
  v38 = v7;
  v40 = a1;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v2 = sub_9AC470(v5, &v41, 0);
  if ( !(_BYTE)v2 )
    return 0;
  v31 = (__int64 *)sub_BD5C60(a1);
  while ( 1 )
  {
    while ( *(_BYTE *)v4 == 42 )
    {
      v14 = *(_QWORD *)(v4 - 64);
      if ( !v14 )
        goto LABEL_12;
      v15 = *(_QWORD *)(v4 - 32);
      if ( *(_BYTE *)v15 != 17 )
        goto LABEL_12;
      v37 = v14;
      v41.m128i_i32[2] = *(_DWORD *)(v38 + 32);
      if ( v41.m128i_i32[2] > 0x40u )
      {
        v30 = v14;
        sub_C43780((__int64)&v41, (const void **)(v38 + 24));
        v14 = v30;
      }
      else
      {
        v41.m128i_i64[0] = *(_QWORD *)(v38 + 24);
      }
      v28 = v14;
      sub_C45EE0((__int64)&v41, (__int64 *)(v15 + 24));
      v36 = v41.m128i_u32[2];
      v35 = v41.m128i_i64[0];
      v16 = sub_ACCFD0(v31, (__int64)&v35);
      v17 = v28;
      v38 = v16;
      if ( v36 > 0x40 && v35 )
      {
        j_j___libc_free_0_0(v35);
        v17 = v28;
      }
      v4 = v17;
    }
    if ( *(_BYTE *)v4 != 58 )
      break;
    v18 = *(_QWORD *)(v4 - 64);
    if ( !v18 )
      break;
    v19 = *(_QWORD *)(v4 - 32);
    if ( *(_BYTE *)v19 != 17 )
      break;
    sub_9AC3E0((__int64)&v41, v18, v32, 0, 0, 0, 0, 1);
    v29 = (const void **)(v19 + 24);
    v34 = *(_DWORD *)(v19 + 32);
    if ( v34 <= 0x40 )
    {
      v20 = *(_QWORD *)(v19 + 24);
      v21 = v20;
LABEL_26:
      v22 = v41.m128i_i64[0] & v20;
LABEL_27:
      v23 = v21 == v22;
      goto LABEL_28;
    }
    sub_C43780((__int64)&v33, v29);
    if ( v34 <= 0x40 )
    {
      v20 = v33;
      v21 = *(_QWORD *)(v19 + 24);
      goto LABEL_26;
    }
    sub_C43B90(&v33, v41.m128i_i64);
    v24 = v34;
    v22 = v33;
    v34 = 0;
    v36 = v24;
    v35 = v33;
    if ( v24 <= 0x40 )
    {
      v21 = *(_QWORD *)(v19 + 24);
      goto LABEL_27;
    }
    v27 = v33;
    v23 = sub_C43C50((__int64)&v35, v29);
    if ( v27 )
    {
      j_j___libc_free_0_0(v27);
      if ( v34 > 0x40 )
      {
        if ( v33 )
          j_j___libc_free_0_0(v33);
      }
    }
LABEL_28:
    if ( v23 )
    {
      v37 = v18;
      v36 = *(_DWORD *)(v38 + 32);
      if ( v36 > 0x40 )
        sub_C43780((__int64)&v35, (const void **)(v38 + 24));
      else
        v35 = *(_QWORD *)(v38 + 24);
      sub_C45EE0((__int64)&v35, (__int64 *)v29);
      v34 = v36;
      v33 = v35;
      v38 = sub_ACCFD0(v31, (__int64)&v33);
      if ( v34 > 0x40 )
      {
        if ( v33 )
          j_j___libc_free_0_0(v33);
      }
    }
    if ( (unsigned int)v43 > 0x40 && v42 )
      j_j___libc_free_0_0(v42);
    if ( v41.m128i_i32[2] > 0x40u && v41.m128i_i64[0] )
      j_j___libc_free_0_0(v41.m128i_u64[0]);
    if ( !v23 )
      break;
    v4 = v37;
  }
LABEL_12:
  v9 = *(unsigned int *)(a2 + 8);
  v10 = (const __m128i *)&v37;
  v11 = *(_QWORD *)a2;
  v12 = v9 + 1;
  if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
  {
    v25 = (const void *)(a2 + 16);
    if ( v11 > (unsigned __int64)&v37 || (unsigned __int64)&v37 >= v11 + 32 * v9 )
    {
      sub_C8D5F0(a2, v25, v12, 0x20u, v12, v8);
      v11 = *(_QWORD *)a2;
      v9 = *(unsigned int *)(a2 + 8);
      v10 = (const __m128i *)&v37;
    }
    else
    {
      v26 = (char *)&v37 - v11;
      sub_C8D5F0(a2, v25, v12, 0x20u, v12, v8);
      v11 = *(_QWORD *)a2;
      v9 = *(unsigned int *)(a2 + 8);
      v10 = (const __m128i *)&v26[*(_QWORD *)a2];
    }
  }
  v13 = (__m128i *)(v11 + 32 * v9);
  *v13 = _mm_loadu_si128(v10);
  v13[1] = _mm_loadu_si128(v10 + 1);
  ++*(_DWORD *)(a2 + 8);
  return v2;
}
