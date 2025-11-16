// Function: sub_B9C270
// Address: 0xb9c270
//
const __m128i **__fastcall sub_B9C270(const __m128i **a1, __int64 a2)
{
  __int64 v3; // rdi
  size_t v4; // rdx
  char v5; // r14
  __int32 v6; // r15d
  unsigned __int8 v7; // al
  __int64 v8; // r8
  __int64 v9; // rax
  size_t v10; // rdx
  __int64 v11; // rax
  const void *v12; // r8
  size_t v13; // rdx
  size_t v14; // rbx
  __int64 *v15; // r13
  const void *v16; // r10
  __m128i v17; // xmm1
  __m128i v18; // xmm2
  size_t v19; // rcx
  __int64 v20; // r15
  __int64 v21; // r14
  __int64 v22; // rsi
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  const void *v27; // [rsp+8h] [rbp-128h]
  char v28; // [rsp+16h] [rbp-11Ah]
  const void *v29; // [rsp+18h] [rbp-118h]
  size_t v30; // [rsp+20h] [rbp-110h]
  __int8 v31; // [rsp+28h] [rbp-108h]
  __int64 v32; // [rsp+30h] [rbp-100h]
  const void *v33; // [rsp+30h] [rbp-100h]
  __int64 v34; // [rsp+38h] [rbp-F8h]
  __int64 v35; // [rsp+40h] [rbp-F0h]
  size_t v36; // [rsp+48h] [rbp-E8h]
  __int64 v37; // [rsp+80h] [rbp-B0h]
  __m128i v38; // [rsp+A0h] [rbp-90h] BYREF
  __m128i v39; // [rsp+B0h] [rbp-80h] BYREF
  __m128i v40; // [rsp+E0h] [rbp-50h]
  __m128i v41; // [rsp+F0h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 40);
  if ( v3 )
  {
    v28 = 1;
    v35 = sub_B91420(v3);
    v36 = v4;
  }
  else
  {
    v28 = 0;
  }
  v5 = *(_BYTE *)(a2 + 32);
  v38 = 0;
  v39 = 0;
  if ( v5 )
  {
    v25 = sub_B91420(*(_QWORD *)(a2 + 24));
    v31 = 1;
    v6 = *(_DWORD *)(a2 + 16);
    v34 = v25;
    v32 = v26;
  }
  else
  {
    v31 = 0;
    v6 = 0;
    v32 = 0;
    v34 = 0;
  }
  v7 = *(_BYTE *)(a2 - 16);
  if ( (v7 & 2) != 0 )
  {
    v8 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
    if ( v8 )
    {
LABEL_7:
      v9 = sub_B91420(v8);
      v30 = v10;
      v8 = v9;
      goto LABEL_8;
    }
  }
  else
  {
    v8 = *(_QWORD *)(a2 - 16 - 8LL * ((v7 >> 2) & 0xF) + 8);
    if ( v8 )
      goto LABEL_7;
  }
  v30 = 0;
LABEL_8:
  v29 = (const void *)v8;
  v11 = sub_A547D0(a2, 0);
  v12 = v29;
  v27 = (const void *)v11;
  v14 = v13;
  v15 = (__int64 *)(*(_QWORD *)(a2 + 8) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a2 + 8) & 4) != 0 )
    v15 = (__int64 *)*v15;
  v38.m128i_i32[0] = v6;
  v16 = (const void *)v35;
  v38.m128i_i64[1] = v34;
  v39.m128i_i64[0] = v32;
  v17 = _mm_loadu_si128(&v38);
  v39.m128i_i8[8] = v31;
  v18 = _mm_loadu_si128(&v39);
  v19 = v36;
  v40 = v17;
  v41 = v18;
  v37 = 0;
  if ( v5 )
  {
    if ( v32 )
    {
      sub_B9B140(v15, (const void *)v40.m128i_i64[1], v41.m128i_u64[0]);
      v19 = v36;
      v16 = (const void *)v35;
      v12 = v29;
    }
    LODWORD(v37) = v6;
  }
  v20 = 0;
  if ( v28 )
  {
    v33 = v12;
    v24 = sub_B9B140(v15, v16, v19);
    v12 = v33;
    v20 = v24;
  }
  v21 = 0;
  if ( v30 )
    v21 = sub_B9B140(v15, v12, v30);
  v22 = 0;
  if ( v14 )
    v22 = sub_B9B140(v15, v27, v14);
  *a1 = sub_B07920(v15, v22, v21, v20, 2u, 1, (const __m128i)0LL, v37);
  return a1;
}
