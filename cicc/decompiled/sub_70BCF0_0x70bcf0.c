// Function: sub_70BCF0
// Address: 0x70bcf0
//
__int16 __fastcall sub_70BCF0(
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
  __m128i v16; // [rsp+0h] [rbp-80h] BYREF
  __int128 v17; // [rsp+10h] [rbp-70h] BYREF
  __m128i v18; // [rsp+20h] [rbp-60h] BYREF
  __m128i v19; // [rsp+30h] [rbp-50h] BYREF
  _OWORD v20[4]; // [rsp+40h] [rbp-40h] BYREF

  *a5 = 0;
  *a6 = 0;
  v10.m128i_i64[0] = sub_709B30(a1, a2);
  v18 = v10;
  v11.m128i_i64[0] = sub_709B30(a1, a3);
  v19 = v11;
  sub_12F9AD0(&v18, &v19, &v17);
  sub_709750(v17, a1, a4, a5, v12);
  v20[0] = _mm_loadu_si128(&v18);
  if ( unk_4F07580 )
  {
    v13 = n - 2;
    if ( (*(_WORD *)((_BYTE *)v20 + n - 2) & 0x7FFF) == 0x7FFF )
    {
LABEL_3:
      *a6 = 1;
      return v13;
    }
    v20[0] = _mm_loadu_si128(&v19);
    v14 = *(_WORD *)((char *)v20 + v13);
    v16 = (__m128i)v20[0];
  }
  else
  {
    LOWORD(v13) = __ROL2__(v20[0], 8) & 0x7FFF;
    if ( (_WORD)v13 == 0x7FFF )
      goto LABEL_3;
    v16 = _mm_loadu_si128(&v19);
    v14 = __ROL2__(v16.m128i_i16[0], 8);
  }
  LODWORD(v13) = v14 & 0x7FFF;
  if ( (_DWORD)v13 == 0x7FFF )
    goto LABEL_3;
  v20[0] = _mm_load_si128(&v16);
  LOWORD(v13) = sub_12F9B10(v20, &unk_4F07870);
  if ( (_BYTE)v13 )
    goto LABEL_3;
  return v13;
}
