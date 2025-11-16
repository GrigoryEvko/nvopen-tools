// Function: sub_29F38C0
// Address: 0x29f38c0
//
__int64 __fastcall sub_29F38C0(__int64 **a1, _BYTE *a2, size_t a3)
{
  __int64 result; // rax
  unsigned __int64 v6; // rax
  _QWORD *v7; // rdx
  __m128i *v8; // rax
  __int64 *v9; // rdi
  _QWORD *v10; // rdi
  _QWORD v11[4]; // [rsp+10h] [rbp-A0h] BYREF
  unsigned __int64 v12[2]; // [rsp+30h] [rbp-80h] BYREF
  __m128i v13; // [rsp+40h] [rbp-70h] BYREF
  unsigned __int64 v14[2]; // [rsp+50h] [rbp-60h] BYREF
  _QWORD v15[2]; // [rsp+60h] [rbp-50h] BYREF
  __int16 v16; // [rsp+70h] [rbp-40h]

  if ( !sub_BA91D0((__int64)a1, a2, a3) )
  {
    sub_BA93D0(a1, 4u, a2, a3, 1u);
    return 0;
  }
  result = (unsigned __int8)qword_50095E8;
  if ( (_BYTE)qword_50095E8 )
    return result;
  v6 = a3;
  v14[0] = (unsigned __int64)v15;
  if ( &a2[a3] && !a2 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v12[0] = a3;
  if ( a3 > 0xF )
  {
    v14[0] = sub_22409D0((__int64)v14, v12, 0);
    v10 = (_QWORD *)v14[0];
    v15[0] = v12[0];
    goto LABEL_18;
  }
  if ( a3 != 1 )
  {
    if ( !a3 )
    {
      v7 = v15;
      goto LABEL_8;
    }
    v10 = v15;
LABEL_18:
    memcpy(v10, a2, a3);
    v6 = v12[0];
    v7 = (_QWORD *)v14[0];
    goto LABEL_8;
  }
  LOBYTE(v15[0]) = *a2;
  v7 = v15;
LABEL_8:
  v14[1] = v6;
  *((_BYTE *)v7 + v6) = 0;
  v8 = (__m128i *)sub_2241130(v14, 0, 0, "Redundant instrumentation detected, with module flag: ", 0x36u);
  v12[0] = (unsigned __int64)&v13;
  if ( (__m128i *)v8->m128i_i64[0] == &v8[1] )
  {
    v13 = _mm_loadu_si128(v8 + 1);
  }
  else
  {
    v12[0] = v8->m128i_i64[0];
    v13.m128i_i64[0] = v8[1].m128i_i64[0];
  }
  v12[1] = v8->m128i_u64[1];
  v8->m128i_i64[0] = (__int64)v8[1].m128i_i64;
  v8->m128i_i64[1] = 0;
  v8[1].m128i_i8[0] = 0;
  if ( (_QWORD *)v14[0] != v15 )
    j_j___libc_free_0(v14[0]);
  v9 = *a1;
  v11[2] = v14;
  v16 = 260;
  v14[0] = (unsigned __int64)v12;
  v11[1] = 0x10000000ALL;
  v11[0] = &unk_49D9EB8;
  sub_B6EB20((__int64)v9, (__int64)v11);
  if ( (__m128i *)v12[0] != &v13 )
    j_j___libc_free_0(v12[0]);
  return 1;
}
