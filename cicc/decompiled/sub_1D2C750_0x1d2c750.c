// Function: sub_1D2C750
// Address: 0x1d2c750
//
__int64 __fastcall sub_1D2C750(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int128 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12,
        int a13,
        __int16 a14,
        __int64 a15)
{
  __int16 v18; // bx
  __int64 v19; // rbx
  int v20; // edx
  __int64 v21; // rax
  __int64 v24; // [rsp+10h] [rbp-60h]
  unsigned __int16 v25; // [rsp+1Eh] [rbp-52h]
  __m128i v26; // [rsp+20h] [rbp-50h] BYREF
  __int64 v27; // [rsp+30h] [rbp-40h]

  v18 = a14;
  v24 = a2;
  if ( !a13 )
  {
    a2 = (unsigned int)a11;
    a13 = sub_1D172F0((__int64)a1, (unsigned int)a11, a12);
  }
  v25 = v18 | 2;
  if ( (a9 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
  {
    a2 = (__int64)&a9;
    sub_1D13370(&v26, (const __m128i *)&a9, (__int64)a1, a7, 0);
    a9 = (__int128)_mm_loadu_si128(&v26);
    a10 = v27;
  }
  v19 = a1[4];
  if ( (_BYTE)a11 )
    v20 = sub_1D13440(a11);
  else
    v20 = sub_1F58D40(&a11, a2, a3, a4, a5, a6);
  v21 = sub_1E0B8E0(v19, v25, (unsigned int)(v20 + 7) >> 3, a13, a15, 0, a9, a10, 1, 0, 0);
  return sub_1D2C2D0(a1, v24, a3, a4, a5, a6, a7, a8, a11, a12, v21);
}
