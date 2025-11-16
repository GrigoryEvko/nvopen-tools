// Function: sub_70B8D0
// Address: 0x70b8d0
//
__int16 __fastcall sub_70B8D0(
        unsigned __int8 a1,
        const __m128i *a2,
        const __m128i *a3,
        _OWORD *a4,
        _DWORD *a5,
        _DWORD *a6)
{
  __m128i v10; // rax
  __m128i v11; // rax
  __int64 v12; // r9
  size_t v13; // rax
  __int16 v14; // ax
  __int128 v16; // [rsp+0h] [rbp-70h] BYREF
  __m128i v17; // [rsp+10h] [rbp-60h] BYREF
  __m128i v18; // [rsp+20h] [rbp-50h] BYREF
  _OWORD v19[4]; // [rsp+30h] [rbp-40h]

  *a5 = 0;
  *a6 = 0;
  v10.m128i_i64[0] = sub_709B30(a1, a2);
  v17 = v10;
  v11.m128i_i64[0] = sub_709B30(a1, a3);
  v18 = v11;
  sub_12F99B0(&v17, &v18, &v16);
  sub_709750(v16, a1, a4, a5, v12);
  v19[0] = _mm_loadu_si128(&v17);
  if ( !unk_4F07580 )
  {
    LOWORD(v13) = __ROL2__(v19[0], 8) & 0x7FFF;
    if ( (_WORD)v13 == 0x7FFF )
      goto LABEL_3;
    v14 = __ROL2__(v18.m128i_i16[0], 8);
LABEL_6:
    LODWORD(v13) = v14 & 0x7FFF;
    if ( (_DWORD)v13 != 0x7FFF )
      return v13;
    goto LABEL_3;
  }
  v13 = n - 2;
  if ( (*(_WORD *)((_BYTE *)v19 + n - 2) & 0x7FFF) != 0x7FFF )
  {
    v19[0] = _mm_loadu_si128(&v18);
    v14 = *(_WORD *)((char *)v19 + v13);
    goto LABEL_6;
  }
LABEL_3:
  *a6 = 1;
  return v13;
}
