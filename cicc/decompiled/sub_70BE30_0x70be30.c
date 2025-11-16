// Function: sub_70BE30
// Address: 0x70be30
//
__int64 __fastcall sub_70BE30(unsigned __int8 a1, const __m128i *a2, const __m128i *a3, _DWORD *a4)
{
  __m128i v6; // rax
  __m128i v7; // rax
  __m128i v8; // xmm0
  __int8 *v9; // rax
  __int8 *v11; // rax
  __m128i v12; // [rsp+0h] [rbp-50h] BYREF
  __m128i v13; // [rsp+10h] [rbp-40h] BYREF
  __m128i v14; // [rsp+20h] [rbp-30h] BYREF

  v6.m128i_i64[0] = sub_709B30(a1, a2);
  v12 = v6;
  v7.m128i_i64[0] = sub_709B30(a1, a3);
  *a4 = 0;
  v8 = _mm_loadu_si128(&v12);
  v13 = v7;
  v14 = v8;
  if ( !unk_4F07580 )
  {
    if ( (__ROL2__(v14.m128i_i16[0], 8) & 0x7FFF) != 0x7FFF )
    {
      v14 = v7;
LABEL_11:
      if ( (__ROL2__(v14.m128i_i16[0], 8) & 0x7FFF) == 0x7FFF )
        goto LABEL_5;
      goto LABEL_12;
    }
LABEL_14:
    v11 = &v14.m128i_i8[2];
    while ( !*v11 )
    {
      if ( (char *)&v14.m128i_u64[1] + 6 == ++v11 )
      {
        v14 = _mm_loadu_si128(&v13);
        if ( unk_4F07580 )
          goto LABEL_4;
        goto LABEL_11;
      }
    }
    goto LABEL_8;
  }
  if ( (v14.m128i_i16[7] & 0x7FFF) == 0x7FFF )
    goto LABEL_14;
  v14 = v7;
LABEL_4:
  if ( (v14.m128i_i16[7] & 0x7FFF) == 0x7FFF )
  {
LABEL_5:
    v9 = &v14.m128i_i8[2];
    while ( !*v9 )
    {
      if ( (char *)&v14.m128i_u64[1] + 6 == ++v9 )
        goto LABEL_12;
    }
LABEL_8:
    *a4 = 1;
    return 0;
  }
LABEL_12:
  if ( (unsigned __int8)((__int64 (__fastcall *)(__m128i *, __m128i *))sub_12F9B30)(&v12, &v13) )
    return (unsigned int)-(unsigned __int8)sub_12F9B50(&v12, &v13);
  else
    return 1;
}
