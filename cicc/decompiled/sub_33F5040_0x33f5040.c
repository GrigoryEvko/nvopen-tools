// Function: sub_33F5040
// Address: 0x33f5040
//
__m128i *__fastcall sub_33F5040(
        _QWORD *a1,
        unsigned __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        unsigned __int64 a6,
        unsigned __int64 a7,
        unsigned __int64 a8,
        __int128 a9,
        __int64 a10,
        __int64 a11,
        unsigned __int64 a12,
        unsigned __int8 a13,
        __int16 a14,
        __int64 a15)
{
  __int64 v18; // rax
  _QWORD *v19; // r11
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // rdx
  unsigned __int64 v24; // rcx
  int v25; // edx
  const __m128i *v26; // rax
  __m128i v28; // xmm0
  _QWORD *v29; // [rsp+8h] [rbp-68h]
  unsigned __int16 v31; // [rsp+1Eh] [rbp-52h]
  __m128i v32; // [rsp+20h] [rbp-50h] BYREF
  int v33; // [rsp+30h] [rbp-40h]
  char v34; // [rsp+34h] [rbp-3Ch]

  v31 = a14 | 2;
  if ( (a9 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    LOWORD(v18) = a11;
    v19 = (_QWORD *)a1[5];
    if ( !(_WORD)a11 )
    {
LABEL_3:
      v29 = v19;
      v20 = sub_3007260((__int64)&a11);
      v19 = v29;
      v21 = v20;
      v18 = v22;
      v32.m128i_i64[0] = v21;
      v23 = v21;
      v32.m128i_i64[1] = v18;
      goto LABEL_4;
    }
  }
  else
  {
    sub_33C8B50((__int64)&v32, (const __m128i *)&a9, (__int64)a1, a7, 0);
    v28 = _mm_loadu_si128(&v32);
    v19 = (_QWORD *)a1[5];
    LODWORD(a10) = v33;
    a9 = (__int128)v28;
    BYTE4(a10) = v34;
    LOWORD(v18) = a11;
    if ( !(_WORD)a11 )
      goto LABEL_3;
  }
  if ( (_WORD)v18 == 1 || (unsigned __int16)(v18 - 504) <= 7u )
    BUG();
  v23 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v18 - 16];
  LOBYTE(v18) = byte_444C4A0[16 * (unsigned __int16)v18 - 8];
LABEL_4:
  v24 = (unsigned __int64)(v23 + 7) >> 3;
  v25 = v24;
  if ( !(_BYTE)v18 )
    v25 = v24;
  v26 = (const __m128i *)sub_2E7BD70(v19, v31, v25, a13, a15, 0, a9, a10, 1u, 0, 0);
  return sub_33F49B0(a1, a2, a3, a4, a5, a6, a7, a8, a11, a12, v26);
}
