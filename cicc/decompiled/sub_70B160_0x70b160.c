// Function: sub_70B160
// Address: 0x70b160
//
char *__fastcall sub_70B160(unsigned __int8 a1, const __m128i *a2, _DWORD *a3, _DWORD *a4, _DWORD *a5)
{
  __m128i v8; // rax
  __int16 v9; // ax
  __int16 v10; // ax
  __int8 *v12; // rax
  int v13; // eax
  __m128i v14; // [rsp+10h] [rbp-50h] BYREF
  __m128i v15; // [rsp+20h] [rbp-40h] BYREF

  if ( a3 )
    *a3 = 0;
  if ( a4 )
    *a4 = 0;
  if ( a5 )
    *a5 = 0;
  v8.m128i_i64[0] = sub_709B30(a1, a2);
  v14 = v8;
  v15 = v8;
  if ( unk_4F07580 )
  {
    if ( (v15.m128i_i16[7] & 0x7FFF) != 0x7FFF )
    {
LABEL_9:
      v9 = *(__int16 *)((char *)&v14.m128i_i16[7] + n);
      goto LABEL_12;
    }
LABEL_17:
    v12 = &v15.m128i_i8[2];
    while ( !*v12 )
    {
      if ( (char *)&v15.m128i_u64[1] + 6 == ++v12 )
      {
        v15 = _mm_loadu_si128(&v14);
        if ( unk_4F07580 )
          goto LABEL_9;
        v10 = v15.m128i_i16[0];
        goto LABEL_11;
      }
    }
    goto LABEL_28;
  }
  v10 = v15.m128i_i16[0];
  if ( (__ROL2__(v15.m128i_i16[0], 8) & 0x7FFF) == 0x7FFF )
    goto LABEL_17;
LABEL_11:
  v9 = __ROL2__(v10, 8);
LABEL_12:
  if ( (v9 & 0x7FFF) == 0x7FFF )
  {
    if ( !(unsigned __int8)sub_12F9B50(&v14, &unk_4F07870) )
      goto LABEL_14;
LABEL_22:
    strcpy(qword_4F07820, "-Infinity");
    if ( a4 )
      *a4 = 1;
    return qword_4F07820;
  }
  if ( (unsigned __int8)sub_12F9B10(&v14, &unk_4F07870) && memcmp(&v14, &unk_4F07870, n) )
  {
    strcpy(qword_4F07820, "-0.0");
    return qword_4F07820;
  }
  if ( (unsigned __int8)(a1 - 9) > 1u && a1 > 1u )
  {
    if ( a1 == 2 || a1 == 11 )
    {
      v13 = sub_8F6180(qword_4F07820);
      goto LABEL_25;
    }
    if ( (unsigned __int8)(a1 - 3) > 1u )
    {
      if ( (unsigned __int8)(a1 - 5) <= 1u )
      {
        if ( !dword_4F07890 )
          goto LABEL_50;
      }
      else
      {
        if ( a1 == 14 || a1 <= 8u )
        {
          if ( a1 == 7 )
          {
            v13 = sub_8F80A0(qword_4F07820);
            goto LABEL_25;
          }
LABEL_40:
          if ( a1 == 8 || a1 == 13 )
          {
            v13 = sub_8F8850(qword_4F07820);
            goto LABEL_25;
          }
LABEL_50:
          v13 = sub_8F8080(qword_4F07820);
          goto LABEL_25;
        }
        if ( qword_4D040A0[a1] != 8 )
          goto LABEL_40;
      }
    }
    v13 = sub_8F6930(qword_4F07820);
    goto LABEL_25;
  }
  v13 = sub_8F59D0(qword_4F07820);
LABEL_25:
  if ( v13 == 2 )
  {
LABEL_14:
    strcpy(qword_4F07820, "+Infinity");
    if ( a3 )
      *a3 = 1;
    return qword_4F07820;
  }
  if ( v13 == 3 )
    goto LABEL_22;
  if ( v13 != 1 )
    return qword_4F07820;
LABEL_28:
  *(_DWORD *)qword_4F07820 = (_DWORD)&loc_4E614E;
  if ( a5 )
    *a5 = 1;
  return qword_4F07820;
}
