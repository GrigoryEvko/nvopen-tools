// Function: sub_CA2060
// Address: 0xca2060
//
__int64 __fastcall sub_CA2060(__int64 a1, const __m128i *a2, bool *a3)
{
  bool v4; // zf
  unsigned int v5; // ebx
  _OWORD v7[2]; // [rsp+10h] [rbp-170h] BYREF
  __int64 v8; // [rsp+30h] [rbp-150h]
  __int128 v9; // [rsp+40h] [rbp-140h] BYREF
  __int64 v10; // [rsp+50h] [rbp-130h]
  _BYTE v11[296]; // [rsp+58h] [rbp-128h] BYREF

  v4 = *(_BYTE *)(a1 + 328) == 0;
  *(_QWORD *)&v9 = v11;
  *((_QWORD *)&v9 + 1) = 0;
  v10 = 256;
  if ( v4 || (*(_BYTE *)(a1 + 320) & 1) != 0 )
  {
    v8 = a2[2].m128i_i64[0];
    v7[0] = _mm_loadu_si128(a2);
    v7[1] = _mm_loadu_si128(a2 + 1);
  }
  else
  {
    sub_CA0EC0((__int64)a2, (__int64)&v9);
    LOWORD(v8) = 261;
    v7[0] = *(_OWORD *)(a1 + 168);
    sub_C846B0((__int64)v7, (unsigned __int8 **)&v9);
    LOWORD(v8) = 261;
    v7[0] = v9;
  }
  v5 = sub_C824F0((__int64)v7, a3);
  if ( (_BYTE *)v9 != v11 )
    _libc_free(v9, a3);
  return v5;
}
