// Function: sub_B18290
// Address: 0xb18290
//
__int64 __fastcall sub_B18290(__int64 a1, __int8 *a2, size_t a3)
{
  __int64 v5; // rdx
  int v6; // eax
  __int64 *v7; // rbx
  __int64 result; // rax
  __int64 v9; // r15
  __m128i *v10; // rbx
  __int64 *m128i_i64; // rcx
  __int64 *v12; // rcx
  size_t v13; // r9
  _BYTE *m128i_i8; // rdi
  int v15; // r13d
  __int64 v16; // rdi
  __int64 v17; // rax
  __m128i *v18; // [rsp+8h] [rbp-58h]
  __int64 *v19; // [rsp+10h] [rbp-50h]
  __int64 v20; // [rsp+18h] [rbp-48h]
  __int64 v21; // [rsp+20h] [rbp-40h] BYREF
  size_t v22[7]; // [rsp+28h] [rbp-38h] BYREF

  v5 = *(unsigned int *)(a1 + 88);
  if ( *(_DWORD *)(a1 + 92) > (unsigned int)v5 )
  {
    v6 = *(_DWORD *)(a1 + 88);
    v7 = (__int64 *)(*(_QWORD *)(a1 + 80) + 80 * v5);
    if ( v7 )
    {
      *v7 = (__int64)(v7 + 2);
      sub_B14B30(v7, "String", (__int64)"");
      v7[4] = (__int64)(v7 + 6);
      sub_B14B30(v7 + 4, a2, (__int64)&a2[a3]);
      v7[8] = 0;
      v7[9] = 0;
      v6 = *(_DWORD *)(a1 + 88);
    }
    result = (unsigned int)(v6 + 1);
    *(_DWORD *)(a1 + 88) = result;
    return result;
  }
  v9 = a1 + 80;
  v20 = a1 + 96;
  v10 = (__m128i *)sub_C8D7D0(a1 + 80, a1 + 96, 0, 80, &v21);
  m128i_i64 = v10[5 * *(unsigned int *)(a1 + 88)].m128i_i64;
  if ( m128i_i64 )
  {
    v18 = &v10[5 * *(unsigned int *)(a1 + 88)];
    *m128i_i64 = (__int64)(m128i_i64 + 2);
    sub_B14B30(m128i_i64, "String", (__int64)"");
    v12 = (__int64 *)v18;
    v13 = a3;
    m128i_i8 = v18[3].m128i_i8;
    v18[2].m128i_i64[0] = (__int64)v18[3].m128i_i64;
    if ( &a2[a3] && !a2 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v22[0] = a3;
    if ( a3 > 0xF )
    {
      v17 = sub_22409D0(&v18[2], v22, 0);
      v12 = (__int64 *)v18;
      m128i_i8 = (_BYTE *)v17;
      v18[2].m128i_i64[0] = v17;
      v18[3].m128i_i64[0] = v22[0];
    }
    else
    {
      if ( a3 == 1 )
      {
        v18[3].m128i_i8[0] = *a2;
LABEL_12:
        v12[5] = v13;
        m128i_i8[v13] = 0;
        v12[8] = 0;
        v12[9] = 0;
        goto LABEL_13;
      }
      if ( !a3 )
        goto LABEL_12;
    }
    v19 = v12;
    memcpy(m128i_i8, a2, a3);
    v12 = v19;
    v13 = v22[0];
    m128i_i8 = (_BYTE *)v19[4];
    goto LABEL_12;
  }
LABEL_13:
  result = sub_B17F60(v9, v10);
  v15 = v21;
  v16 = *(_QWORD *)(a1 + 80);
  if ( v20 != v16 )
    result = _libc_free(v16, v10);
  ++*(_DWORD *)(a1 + 88);
  *(_QWORD *)(a1 + 80) = v10;
  *(_DWORD *)(a1 + 92) = v15;
  return result;
}
