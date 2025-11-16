// Function: sub_D87C70
// Address: 0xd87c70
//
__m128i *__fastcall sub_D87C70(_QWORD *a1, const __m128i *a2, __int64 a3)
{
  __m128i *v5; // r14
  __int64 *m128i_i64; // r13
  unsigned int v7; // eax
  unsigned int v8; // eax
  __int64 *v9; // r8
  __int64 v10; // rax
  __int64 v11; // r12
  _QWORD *v12; // r9
  unsigned __int64 v13; // r15
  __int64 v14; // rax
  char v15; // cl
  unsigned __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rdi
  __int64 v20; // rax
  __int64 *v21; // [rsp+0h] [rbp-40h]
  _QWORD *v22; // [rsp+8h] [rbp-38h]

  v5 = (__m128i *)sub_22077B0(80);
  m128i_i64 = v5[3].m128i_i64;
  v5[2] = _mm_loadu_si128(a2);
  v7 = *(_DWORD *)(a3 + 8);
  v5[3].m128i_i32[2] = v7;
  if ( v7 > 0x40 )
    sub_C43780((__int64)m128i_i64, (const void **)a3);
  else
    v5[3].m128i_i64[0] = *(_QWORD *)a3;
  v8 = *(_DWORD *)(a3 + 24);
  v9 = v5[4].m128i_i64;
  v5[4].m128i_i32[2] = v8;
  if ( v8 > 0x40 )
  {
    sub_C43780((__int64)v5[4].m128i_i64, (const void **)(a3 + 16));
    v11 = a1[2];
    v9 = v5[4].m128i_i64;
    v12 = a1 + 1;
    if ( v11 )
      goto LABEL_5;
LABEL_24:
    v11 = (__int64)v12;
    if ( (_QWORD *)a1[3] != v12 )
    {
      v13 = v5[2].m128i_u64[1];
      goto LABEL_16;
    }
    goto LABEL_28;
  }
  v10 = *(_QWORD *)(a3 + 16);
  v11 = a1[2];
  v12 = a1 + 1;
  v5[4].m128i_i64[0] = v10;
  if ( !v11 )
    goto LABEL_24;
LABEL_5:
  v13 = v5[2].m128i_u64[1];
  while ( 1 )
  {
    v16 = *(_QWORD *)(v11 + 40);
    if ( *(_OWORD *)&v5[2] < *(_OWORD *)(v11 + 32) )
      break;
    v14 = *(_QWORD *)(v11 + 24);
    v15 = 0;
    if ( !v14 )
      goto LABEL_11;
LABEL_8:
    v11 = v14;
  }
  v14 = *(_QWORD *)(v11 + 16);
  v15 = 1;
  if ( v14 )
    goto LABEL_8;
LABEL_11:
  if ( !v15 )
  {
    v17 = v11;
    if ( v16 < v13 )
      goto LABEL_13;
LABEL_18:
    if ( v13 == v16 && *(_QWORD *)(v11 + 32) < v5[2].m128i_i64[0] )
    {
      v11 = v17;
      goto LABEL_21;
    }
    goto LABEL_22;
  }
  if ( v11 == a1[3] )
    goto LABEL_13;
LABEL_16:
  v21 = v9;
  v22 = v12;
  v20 = sub_220EF80(v11);
  v12 = v22;
  v9 = v21;
  v16 = *(_QWORD *)(v20 + 40);
  if ( v16 >= v13 )
  {
    v17 = v11;
    v11 = v20;
    goto LABEL_18;
  }
LABEL_21:
  if ( v11 )
  {
LABEL_13:
    v18 = 1;
    if ( v12 == (_QWORD *)v11 )
    {
LABEL_14:
      sub_220F040(v18, v5, v11, v12);
      ++a1[5];
      return v5;
    }
    if ( __PAIR128__(v13, v5[2].m128i_i64[0]) >= *(_OWORD *)(v11 + 32) )
    {
      v18 = 0;
      goto LABEL_14;
    }
LABEL_28:
    v18 = 1;
    goto LABEL_14;
  }
LABEL_22:
  sub_969240(v9);
  sub_969240(m128i_i64);
  j_j___libc_free_0(v5, 80);
  return (__m128i *)v11;
}
