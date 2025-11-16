// Function: sub_2EF8D50
// Address: 0x2ef8d50
//
_QWORD *__fastcall sub_2EF8D50(__int64 a1)
{
  __int8 *v1; // r13
  size_t v2; // r12
  __m128i *v3; // rax
  _QWORD *v4; // rax
  _QWORD *v5; // r12
  __m128i *v6; // rax
  size_t v7; // rax
  __int128 *v8; // rax
  __m128i *v10; // rdi
  __int64 v11; // [rsp+8h] [rbp-48h] BYREF
  __m128i *v12; // [rsp+10h] [rbp-40h] BYREF
  size_t v13; // [rsp+18h] [rbp-38h]
  __m128i v14[3]; // [rsp+20h] [rbp-30h] BYREF

  v1 = *(__int8 **)a1;
  v2 = *(_QWORD *)(a1 + 8);
  v12 = v14;
  if ( &v1[v2] && !v1 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v11 = v2;
  if ( v2 > 0xF )
  {
    v12 = (__m128i *)sub_22409D0((__int64)&v12, (unsigned __int64 *)&v11, 0);
    v10 = v12;
    v14[0].m128i_i64[0] = v11;
  }
  else
  {
    if ( v2 == 1 )
    {
      v14[0].m128i_i8[0] = *v1;
      v3 = v14;
      goto LABEL_6;
    }
    if ( !v2 )
    {
      v3 = v14;
      goto LABEL_6;
    }
    v10 = v14;
  }
  memcpy(v10, v1, v2);
  v2 = v11;
  v3 = v12;
LABEL_6:
  v13 = v2;
  v3->m128i_i8[v2] = 0;
  v4 = (_QWORD *)sub_22077B0(0xE8u);
  v5 = v4;
  if ( v4 )
  {
    v4[1] = 0;
    v4[2] = &unk_502235C;
    v4[7] = v4 + 13;
    v4[14] = v4 + 20;
    *v4 = off_4A2A3B8;
    v4[25] = v4 + 27;
    v6 = v12;
    *((_DWORD *)v5 + 6) = 2;
    v5[4] = 0;
    v5[5] = 0;
    v5[6] = 0;
    v5[8] = 1;
    v5[9] = 0;
    v5[10] = 0;
    v5[12] = 0;
    v5[13] = 0;
    v5[15] = 1;
    v5[16] = 0;
    v5[17] = 0;
    v5[19] = 0;
    v5[20] = 0;
    *((_BYTE *)v5 + 168) = 0;
    v5[22] = 0;
    v5[23] = 0;
    v5[24] = 0;
    *((_DWORD *)v5 + 22) = 1065353216;
    *((_DWORD *)v5 + 36) = 1065353216;
    if ( v6 == v14 )
    {
      *(__m128i *)(v5 + 27) = _mm_load_si128(v14);
    }
    else
    {
      v5[25] = v6;
      v5[27] = v14[0].m128i_i64[0];
    }
    v7 = v13;
    v12 = v14;
    v13 = 0;
    v5[26] = v7;
    v14[0].m128i_i8[0] = 0;
    v8 = sub_BC2B00();
    sub_2EF8B00((__int64)v8);
  }
  if ( v12 != v14 )
    j_j___libc_free_0((unsigned __int64)v12);
  return v5;
}
