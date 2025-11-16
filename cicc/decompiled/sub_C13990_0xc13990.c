// Function: sub_C13990
// Address: 0xc13990
//
__int64 __fastcall sub_C13990(_QWORD *a1, __int8 *a2, size_t a3, __int32 a4)
{
  _QWORD *v4; // r8
  size_t v5; // rax
  char **v8; // r13
  __m128i *v9; // rdx
  _QWORD *v10; // rdi
  __int64 v11; // rdx
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 result; // rax
  char *v16; // rsi
  __m128i *v17; // rdi
  __int64 v18; // rax
  __m128i *v19; // rdi
  _QWORD *v20; // [rsp+8h] [rbp-68h]
  __int64 v21; // [rsp+18h] [rbp-58h] BYREF
  __m128i *v22; // [rsp+20h] [rbp-50h] BYREF
  size_t v23; // [rsp+28h] [rbp-48h]
  __m128i v24[4]; // [rsp+30h] [rbp-40h] BYREF

  v4 = a1;
  v5 = a3;
  v22 = v24;
  v8 = (char **)*a1;
  if ( &a2[a3] && !a2 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v21 = a3;
  if ( a3 > 0xF )
  {
    v18 = sub_22409D0(&v22, &v21, 0);
    v4 = a1;
    v22 = (__m128i *)v18;
    v19 = (__m128i *)v18;
    v24[0].m128i_i64[0] = v21;
    goto LABEL_21;
  }
  if ( a3 != 1 )
  {
    if ( !a3 )
    {
      v9 = v24;
      goto LABEL_6;
    }
    v19 = v24;
LABEL_21:
    v20 = v4;
    memcpy(v19, a2, a3);
    v5 = v21;
    v9 = v22;
    v4 = v20;
    goto LABEL_6;
  }
  v24[0].m128i_i8[0] = *a2;
  v9 = v24;
LABEL_6:
  v23 = v5;
  v9->m128i_i8[v5] = 0;
  v10 = (_QWORD *)*v4;
  v11 = *(_QWORD *)(*v4 + 8LL);
  v10[11] += 40LL;
  v12 = (v11 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v10[2] >= v12 + 40 && v11 )
  {
    v10[1] = v12 + 40;
    v13 = (v11 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  }
  else
  {
    v13 = sub_9D1E70((__int64)(v10 + 1), 40, 40, 3);
    v12 = v13 & 0xFFFFFFFFFFFFFFFBLL;
  }
  *(_QWORD *)v13 = v13 + 16;
  if ( v22 == v24 )
  {
    *(__m128i *)(v13 + 16) = _mm_load_si128(v24);
  }
  else
  {
    *(_QWORD *)v13 = v22;
    *(_QWORD *)(v13 + 16) = v24[0].m128i_i64[0];
  }
  v14 = v23;
  result = v12 | 4;
  *(_DWORD *)(v13 + 32) = a4;
  v22 = v24;
  *(_QWORD *)(v13 + 8) = v14;
  v21 = result;
  v16 = v8[14];
  v23 = 0;
  v24[0].m128i_i8[0] = 0;
  if ( v16 == v8[15] )
  {
    result = sub_C13810(v8 + 13, v16, &v21);
    v17 = v22;
  }
  else
  {
    v17 = v24;
    if ( v16 )
    {
      *(_QWORD *)v16 = result;
      v16 = v8[14];
      v17 = v22;
    }
    v8[14] = v16 + 8;
  }
  if ( v17 != v24 )
    return j_j___libc_free_0(v17, v24[0].m128i_i64[0] + 1);
  return result;
}
