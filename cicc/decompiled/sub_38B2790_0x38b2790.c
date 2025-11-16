// Function: sub_38B2790
// Address: 0x38b2790
//
__int64 __fastcall sub_38B2790(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r13d
  __int64 v4; // r8
  __int64 v5; // r9
  __m128i v6; // xmm0
  __int64 *v7; // r14
  _QWORD *v8; // rax
  __int64 v9; // rdx
  unsigned int v10; // [rsp+2Ch] [rbp-64h] BYREF
  __m128i v11; // [rsp+30h] [rbp-60h] BYREF
  int v12[4]; // [rsp+40h] [rbp-50h] BYREF
  _QWORD *v13; // [rsp+50h] [rbp-40h] BYREF
  size_t v14; // [rsp+58h] [rbp-38h]
  _BYTE v15[48]; // [rsp+60h] [rbp-30h] BYREF

  v10 = a2;
  v13 = v15;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  v14 = 0;
  v15[0] = 0;
  if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here")
    || (unsigned __int8)sub_388AF10(a1, 303, "expected 'path' here")
    || (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_388B0A0(a1, (unsigned __int64 *)&v13)
    || (unsigned __int8)sub_388AF10(a1, 4, "expected ',' here")
    || (unsigned __int8)sub_388AF10(a1, 304, "expected 'hash' here")
    || (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here")
    || (unsigned __int8)sub_388BA90(a1, &v11)
    || (unsigned __int8)sub_388AF10(a1, 4, "expected ',' here")
    || (unsigned __int8)sub_388BA90(a1, &v11.m128i_i32[1])
    || (unsigned __int8)sub_388AF10(a1, 4, "expected ',' here")
    || (unsigned __int8)sub_388BA90(a1, &v11.m128i_i32[2])
    || (unsigned __int8)sub_388AF10(a1, 4, "expected ',' here")
    || (unsigned __int8)sub_388BA90(a1, &v11.m128i_i32[3])
    || (unsigned __int8)sub_388AF10(a1, 4, "expected ',' here")
    || (unsigned __int8)sub_388BA90(a1, v12)
    || (unsigned __int8)sub_388AF10(a1, 13, "expected ')' here")
    || (v2 = sub_388AF10(a1, 13, "expected ')' here"), (_BYTE)v2) )
  {
    v2 = 1;
  }
  else
  {
    v6 = _mm_loadu_si128(&v11);
    v7 = (__int64 *)sub_14EC8B0(*(_QWORD *)(a1 + 184), v13, v14, v10, v4, v5, v6.m128i_i64[0], v6.m128i_i64[1], v12[0]);
    v8 = (_QWORD *)sub_38B2690((_QWORD *)(a1 + 1392), &v10);
    v9 = *v7;
    *v8 = v7 + 5;
    v8[1] = v9;
  }
  if ( v13 != (_QWORD *)v15 )
    j_j___libc_free_0((unsigned __int64)v13);
  return v2;
}
