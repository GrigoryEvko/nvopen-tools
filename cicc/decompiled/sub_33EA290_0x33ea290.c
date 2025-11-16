// Function: sub_33EA290
// Address: 0x33ea290
//
__m128i *__fastcall sub_33EA290(
        __int64 *a1,
        int a2,
        char a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8,
        __int64 a9,
        __int128 a10,
        __int128 a11,
        __int64 a12,
        __int64 a13,
        __int64 a14,
        unsigned __int8 a15,
        __int16 a16,
        __int64 a17,
        __int64 a18)
{
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 v25; // rdx
  _QWORD *v26; // rdi
  unsigned __int64 v27; // rcx
  int v28; // edx
  const __m128i *v29; // rax
  __m128i v31; // xmm0
  unsigned __int16 v33; // [rsp+Eh] [rbp-52h]
  __m128i v34; // [rsp+10h] [rbp-50h] BYREF
  int v35; // [rsp+20h] [rbp-40h]
  char v36; // [rsp+24h] [rbp-3Ch]

  v33 = a16 | 1;
  if ( (a11 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    LOWORD(v22) = a13;
    if ( !(_WORD)a13 )
    {
LABEL_3:
      v23 = sub_3007260((__int64)&a13);
      v22 = v24;
      v34.m128i_i64[0] = v23;
      v25 = v23;
      v34.m128i_i64[1] = v22;
      goto LABEL_4;
    }
  }
  else
  {
    sub_33C8C10((__int64)&v34, (const __m128i *)&a11, (__int64)a1, a8, a10);
    v31 = _mm_loadu_si128(&v34);
    LODWORD(a12) = v35;
    a11 = (__int128)v31;
    BYTE4(a12) = v36;
    LOWORD(v22) = a13;
    if ( !(_WORD)a13 )
      goto LABEL_3;
  }
  if ( (_WORD)v22 == 1 || (unsigned __int16)(v22 - 504) <= 7u )
    BUG();
  v25 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v22 - 16];
  LOBYTE(v22) = byte_444C4A0[16 * (unsigned __int16)v22 - 8];
LABEL_4:
  v26 = (_QWORD *)a1[5];
  v27 = (unsigned __int64)(v25 + 7) >> 3;
  v28 = v27;
  if ( !(_BYTE)v22 )
    v28 = v27;
  v29 = (const __m128i *)sub_2E7BD70(v26, v33, v28, a15, a17, a18, a11, a12, 1u, 0, 0);
  return sub_33E9C90(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a13, a14, v29);
}
