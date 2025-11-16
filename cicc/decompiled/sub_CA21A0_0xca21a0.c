// Function: sub_CA21A0
// Address: 0xca21a0
//
__int64 __fastcall sub_CA21A0(__int64 a1, const __m128i *a2, struct passwd *a3, __int64 a4, __int64 a5)
{
  bool v6; // zf
  unsigned int v7; // ebx
  _OWORD v9[2]; // [rsp+10h] [rbp-170h] BYREF
  __int64 v10; // [rsp+30h] [rbp-150h]
  __int128 v11; // [rsp+40h] [rbp-140h] BYREF
  __int64 v12; // [rsp+50h] [rbp-130h]
  _BYTE v13[296]; // [rsp+58h] [rbp-128h] BYREF

  v6 = *(_BYTE *)(a1 + 328) == 0;
  *(_QWORD *)&v11 = v13;
  *((_QWORD *)&v11 + 1) = 0;
  v12 = 256;
  if ( v6 || (*(_BYTE *)(a1 + 320) & 1) != 0 )
  {
    v10 = a2[2].m128i_i64[0];
    v9[0] = _mm_loadu_si128(a2);
    v9[1] = _mm_loadu_si128(a2 + 1);
  }
  else
  {
    sub_CA0EC0((__int64)a2, (__int64)&v11);
    LOWORD(v10) = 261;
    v9[0] = *(_OWORD *)(a1 + 168);
    sub_C846B0((__int64)v9, (unsigned __int8 **)&v11);
    LOWORD(v10) = 261;
    v9[0] = v11;
  }
  v7 = sub_C84130((__int64)v9, a3, 0, a4, a5);
  if ( (_BYTE *)v11 != v13 )
    _libc_free(v11, a3);
  return v7;
}
