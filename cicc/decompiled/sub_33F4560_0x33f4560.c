// Function: sub_33F4560
// Address: 0x33f4560
//
__m128i *__fastcall sub_33F4560(
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
        unsigned __int8 a11,
        __int16 a12,
        __int64 a13)
{
  _QWORD *v16; // r11
  __int64 v17; // rax
  unsigned __int16 v18; // dx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  unsigned __int64 v25; // rcx
  int v26; // edx
  const __m128i *v27; // rax
  __m128i v29; // xmm0
  _QWORD *v30; // [rsp+8h] [rbp-78h]
  unsigned __int16 v32; // [rsp+1Eh] [rbp-62h]
  unsigned __int16 v33; // [rsp+20h] [rbp-60h] BYREF
  __int64 v34; // [rsp+28h] [rbp-58h]
  __m128i v35; // [rsp+30h] [rbp-50h] BYREF
  int v36; // [rsp+40h] [rbp-40h]
  char v37; // [rsp+44h] [rbp-3Ch]

  v32 = a12 | 2;
  if ( (a9 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
  {
    sub_33C8B50((__int64)&v35, (const __m128i *)&a9, (__int64)a1, a7, 0);
    v29 = _mm_loadu_si128(&v35);
    LODWORD(a10) = v36;
    a9 = (__int128)v29;
    BYTE4(a10) = v37;
  }
  v16 = (_QWORD *)a1[5];
  v17 = *(_QWORD *)(a5 + 48) + 16LL * (unsigned int)a6;
  v18 = *(_WORD *)v17;
  v19 = *(_QWORD *)(v17 + 8);
  v33 = v18;
  v34 = v19;
  if ( v18 )
  {
    if ( v18 == 1 || (unsigned __int16)(v18 - 504) <= 7u )
      BUG();
    v23 = 16LL * (v18 - 1) + 71615648;
    v24 = *(_QWORD *)&byte_444C4A0[16 * v18 - 16];
    LOBYTE(v23) = *(_BYTE *)(v23 + 8);
  }
  else
  {
    v30 = v16;
    v20 = sub_3007260((__int64)&v33);
    v16 = v30;
    v21 = v20;
    v23 = v22;
    v35.m128i_i64[0] = v21;
    v24 = v21;
    v35.m128i_i64[1] = v23;
  }
  v25 = (unsigned __int64)(v24 + 7) >> 3;
  v26 = v25;
  if ( !(_BYTE)v23 )
    v26 = v25;
  v27 = (const __m128i *)sub_2E7BD70(v16, v32, v26, a11, a13, 0, a9, a10, 1u, 0, 0);
  return sub_33F3F90(a1, a2, a3, a4, a5, a6, a7, a8, v27);
}
