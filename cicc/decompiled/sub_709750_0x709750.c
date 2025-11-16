// Function: sub_709750
// Address: 0x709750
//
__int64 __fastcall sub_709750(__int128 a1, __int64 a2, _OWORD *a3, _DWORD *a4, __int64 a5)
{
  __int64 result; // rax
  __m128i v8; // xmm1
  unsigned int v9; // r13d
  __int16 v10; // ax
  __m128i si128; // xmm2
  unsigned int v12; // r13d
  __int16 v13; // ax
  __int16 v14; // ax
  __m128i v15; // xmm3
  __int64 v16; // r13
  __int16 v17; // ax
  __int128 src; // [rsp+0h] [rbp-50h] BYREF
  __m128i v19; // [rsp+10h] [rbp-40h] BYREF
  __m128i v20; // [rsp+20h] [rbp-30h]

  result = (unsigned int)*a4;
  src = a1;
  if ( (_DWORD)result )
    return result;
  *a3 = 0;
  if ( (unsigned __int8)a2 <= 1u || (_BYTE)a2 == 10 )
  {
    si128 = _mm_load_si128((const __m128i *)&src);
    unk_4F968EA = 0;
    v19 = si128;
    v12 = ((__int64 (__fastcall *)(__m128i *, _QWORD, __int64, _OWORD *, _DWORD *, __int64, _QWORD, _QWORD))sub_12F9950)(
            &v19,
            *((_QWORD *)&a1 + 1),
            a2,
            a3,
            a4,
            a5,
            src,
            *((_QWORD *)&src + 1));
    result = unk_4F968EA;
    if ( (unk_4F968EA & 4) != 0 )
    {
      result = HIDWORD(qword_4F077B4);
      if ( !HIDWORD(qword_4F077B4) )
        goto LABEL_22;
      v20 = _mm_load_si128(&v19);
      v14 = unk_4F07580 ? *(__int16 *)((char *)&v19.m128i_i16[7] + n) : __ROL2__(v20.m128i_i16[0], 8);
      result = v14 & 0x7FFF;
      if ( (_DWORD)result == 0x7FFF )
        goto LABEL_22;
    }
    else if ( (unk_4F968EA & 2) != 0 )
    {
      result = sub_12F9680(v12, (unsigned __int16)word_4F0786C);
      if ( (_BYTE)result )
        goto LABEL_22;
    }
    if ( !*a4 )
      *(_WORD *)a3 = v12;
  }
  else if ( (a2 & 0xFD) == 9 || (_BYTE)a2 == 2 )
  {
    v8 = _mm_load_si128((const __m128i *)&src);
    unk_4F968EA = 0;
    v19 = v8;
    v9 = sub_12F9960(&v19);
    result = unk_4F968EA;
    if ( (unk_4F968EA & 4) != 0 )
    {
      result = (__int64)&qword_4F077B4 + 4;
      if ( !HIDWORD(qword_4F077B4) )
        goto LABEL_22;
      v20 = _mm_load_si128(&v19);
      v10 = unk_4F07580 ? *(__int16 *)((char *)&v19.m128i_i16[7] + n) : __ROL2__(v20.m128i_i16[0], 8);
      result = v10 & 0x7FFF;
      if ( (_DWORD)result == 0x7FFF )
        goto LABEL_22;
    }
    else if ( (unk_4F968EA & 2) != 0 )
    {
      result = sub_12F9730(v9, (unsigned int)dword_4F07868);
      if ( (_BYTE)result )
        goto LABEL_22;
    }
    if ( !*a4 )
      *(_DWORD *)a3 = v9;
  }
  else
  {
    if ( (unsigned __int8)(a2 - 3) <= 1u )
      goto LABEL_45;
    if ( (unsigned __int8)(a2 - 5) > 1u )
    {
      if ( (_BYTE)a2 == 14 || (unsigned __int8)a2 <= 8u || qword_4D040A0[(unsigned __int8)a2] != 8 )
        return (__int64)memcpy(a3, &src, n);
LABEL_45:
      v15 = _mm_load_si128((const __m128i *)&src);
      unk_4F968EA = 0;
      v19 = v15;
      v16 = sub_12F99A0(&v19);
      result = unk_4F968EA;
      if ( (unk_4F968EA & 4) != 0 )
      {
        result = (__int64)&qword_4F077B4 + 4;
        if ( HIDWORD(qword_4F077B4) )
        {
          v20 = _mm_load_si128(&v19);
          v17 = unk_4F07580 ? *(__int16 *)((char *)&v19.m128i_i16[7] + n) : __ROL2__(v20.m128i_i16[0], 8);
          result = v17 & 0x7FFF;
          if ( (_DWORD)result != 0x7FFF )
            goto LABEL_50;
        }
      }
      else if ( (unk_4F968EA & 2) == 0 || (result = sub_12F97F0(v16, qword_4F07860), !(_BYTE)result) )
      {
LABEL_50:
        if ( !*a4 )
          *(_QWORD *)a3 = v16;
        return result;
      }
LABEL_22:
      *a4 = 1;
      return result;
    }
    if ( dword_4F07890 )
      goto LABEL_45;
    v19 = _mm_load_si128((const __m128i *)&src);
    unk_4F968EA = 0;
    if ( unk_4F06930 == 64 )
    {
      ((void (__fastcall *)(__m128i *, _OWORD *, __int64, _OWORD *, _QWORD, __int64, _QWORD, _QWORD))sub_12F9970)(
        &v19,
        a3,
        a2,
        a3,
        (unsigned int)dword_4F07890,
        a5,
        src,
        *((_QWORD *)&src + 1));
    }
    else
    {
      if ( unk_4F06930 != 113 )
        sub_721090(a1);
      *a3 = _mm_load_si128(&v19);
    }
    v20 = _mm_load_si128(&v19);
    if ( unk_4F07580 )
      v13 = *(__int16 *)((char *)&v19.m128i_i16[7] + n);
    else
      v13 = __ROL2__(v20.m128i_i16[0], 8);
    result = v13 & 0x7FFF;
    if ( (_DWORD)result != 0x7FFF && (unk_4F968EA & 4) != 0 )
    {
      result = HIDWORD(qword_4F077B4);
      if ( !HIDWORD(qword_4F077B4) )
        goto LABEL_22;
    }
  }
  return result;
}
