// Function: sub_26D04C0
// Address: 0x26d04c0
//
void __fastcall sub_26D04C0(__int64 a1, int *a2, __int64 a3)
{
  size_t v3; // r13
  unsigned int v6; // esi
  __int64 v7; // rdi
  int v8; // r10d
  __int64 *v9; // r15
  unsigned int v10; // eax
  __int64 *v11; // rcx
  __int64 v12; // r9
  int v13; // eax
  int v14; // ecx
  __m128i *v15; // rdi
  __int8 *v16; // rdx
  unsigned __int64 v17; // rax
  __int32 v18; // ecx
  __int64 v19; // r12
  _QWORD *v20; // rax
  __int64 v21; // r13
  unsigned __int64 v22; // r12
  _QWORD *v23; // rax
  _QWORD *v24; // rdx
  _QWORD *v25; // rcx
  char v26; // di
  int v27; // eax
  _QWORD *v28; // [rsp+8h] [rbp-E8h]
  __int64 v29[2]; // [rsp+10h] [rbp-E0h] BYREF
  __m128i v30; // [rsp+20h] [rbp-D0h] BYREF
  __int32 v31; // [rsp+38h] [rbp-B8h] BYREF
  unsigned __int64 v32; // [rsp+40h] [rbp-B0h]
  __int32 *v33; // [rsp+48h] [rbp-A8h]
  __int32 *v34; // [rsp+50h] [rbp-A0h]
  __int64 v35; // [rsp+58h] [rbp-98h]

  v3 = a3;
  if ( a2 )
  {
    sub_C7D030(&v30);
    sub_C7D280(v30.m128i_i32, a2, v3);
    sub_C7D290(&v30, v29);
    a3 = v29[0];
  }
  v6 = *(_DWORD *)(a1 + 112);
  v29[0] = a3;
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 88);
    v30.m128i_i64[0] = 0;
LABEL_27:
    v6 *= 2;
    goto LABEL_28;
  }
  v7 = *(_QWORD *)(a1 + 96);
  v8 = 1;
  v9 = 0;
  v10 = (v6 - 1) & (((0xBF58476D1CE4E5B9LL * a3) >> 31) ^ (484763065 * a3));
  v11 = (__int64 *)(v7 + 16LL * v10);
  v12 = *v11;
  if ( a3 == *v11 )
    return;
  while ( v12 != -1 )
  {
    if ( v12 != -2 || v9 )
      v11 = v9;
    v10 = (v6 - 1) & (v8 + v10);
    v12 = *(_QWORD *)(v7 + 16LL * v10);
    if ( a3 == v12 )
      return;
    ++v8;
    v9 = v11;
    v11 = (__int64 *)(v7 + 16LL * v10);
  }
  v13 = *(_DWORD *)(a1 + 104);
  if ( !v9 )
    v9 = v11;
  ++*(_QWORD *)(a1 + 88);
  v14 = v13 + 1;
  v30.m128i_i64[0] = (__int64)v9;
  if ( 4 * (v13 + 1) >= 3 * v6 )
    goto LABEL_27;
  if ( v6 - *(_DWORD *)(a1 + 108) - v14 <= v6 >> 3 )
  {
LABEL_28:
    sub_26D02C0(a1 + 88, v6);
    sub_26C7A50(a1 + 88, v29, &v30);
    a3 = v29[0];
    v9 = (__int64 *)v30.m128i_i64[0];
    v14 = *(_DWORD *)(a1 + 104) + 1;
  }
  *(_DWORD *)(a1 + 104) = v14;
  if ( *v9 != -1 )
    --*(_DWORD *)(a1 + 108);
  *v9 = a3;
  v9[1] = 0;
  v30.m128i_i64[0] = (__int64)a2;
  v30.m128i_i64[1] = v3;
  v31 = 0;
  v32 = 0;
  v33 = &v31;
  v34 = &v31;
  v35 = 0;
  v15 = (__m128i *)sub_22077B0(0x50u);
  v16 = &v15[2].m128i_i8[8];
  v15[1] = _mm_loadu_si128(&v30);
  v17 = v32;
  if ( v32 )
  {
    v18 = v31;
    v15[3].m128i_i64[0] = v32;
    v15[2].m128i_i32[2] = v18;
    v15[3].m128i_i64[1] = (__int64)v33;
    v15[4].m128i_i64[0] = (__int64)v34;
    *(_QWORD *)(v17 + 8) = v16;
    v32 = 0;
    v15[4].m128i_i64[1] = v35;
    v33 = &v31;
    v34 = &v31;
    v35 = 0;
  }
  else
  {
    v15[2].m128i_i32[2] = 0;
    v15[3].m128i_i64[0] = 0;
    v15[3].m128i_i64[1] = (__int64)v16;
    v15[4].m128i_i64[0] = (__int64)v16;
    v15[4].m128i_i64[1] = 0;
  }
  sub_2208C80(v15, a1 + 64);
  ++*(_QWORD *)(a1 + 80);
  v19 = *(_QWORD *)(a1 + 72) + 16LL;
  sub_26BBDA0(v32);
  v9[1] = v19;
  v20 = (_QWORD *)sub_22077B0(0x38u);
  v21 = v9[1];
  v20[4] = a1;
  v22 = (unsigned __int64)v20;
  v20[5] = v21;
  v20[6] = 0;
  v23 = sub_26C4C60(a1 + 16, (__int64)(v20 + 4));
  if ( v24 )
  {
    v25 = (_QWORD *)(a1 + 24);
    v26 = 1;
    if ( !v23 && v24 != v25 )
    {
      v28 = v24;
      v27 = sub_C1F8C0(v21, v24[5]);
      v25 = (_QWORD *)(a1 + 24);
      v24 = v28;
      v26 = v27 < 0;
    }
    sub_220F040(v26, v22, v24, v25);
    ++*(_QWORD *)(a1 + 56);
  }
  else
  {
    j_j___libc_free_0(v22);
  }
}
