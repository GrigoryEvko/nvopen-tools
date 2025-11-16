// Function: sub_C3B580
// Address: 0xc3b580
//
__int64 __fastcall sub_C3B580(__int64 *a1, __int64 *a2, char a3)
{
  float v3; // xmm0_4
  __m128i v4; // xmm0
  float v6; // xmm0_4
  float v7; // [rsp+Ch] [rbp-44h]
  float v8; // [rsp+Ch] [rbp-44h]
  float v9; // [rsp+Ch] [rbp-44h]
  char v10; // [rsp+1Fh] [rbp-31h] BYREF
  _BYTE v11[48]; // [rsp+20h] [rbp-30h] BYREF

  if ( a3 == 2 )
    goto LABEL_13;
  if ( a3 > 2 )
  {
    if ( a3 != 3 )
      return 1;
    v8 = sub_C3AAA0(a2);
    v6 = sub_C3AAA0(a1);
    *(float *)v4.m128i_i32 = sub_CED4B0(&v10, 1, 1, v6, v8);
LABEL_9:
    if ( !(unsigned int)sub_CED540(&v10) )
    {
      v4 = (__m128i)v4.m128i_u32[0];
      goto LABEL_6;
    }
    goto LABEL_11;
  }
  if ( !a3 )
  {
LABEL_13:
    v9 = sub_C3AAA0(a2);
    *(float *)v4.m128i_i32 = sub_C3AAA0(a1);
    sub_CED4A0(&v10, 1, 1, *(float *)v4.m128i_i32, v9);
    goto LABEL_9;
  }
  if ( a3 == 1 )
  {
    v7 = sub_C3AAA0(a2);
    v3 = sub_C3AAA0(a1);
    v4 = (__m128i)COERCE_UNSIGNED_INT(sub_CED490(&v10, 1, 1, v3, v7));
    if ( !(unsigned int)sub_CED540(&v10) )
    {
LABEL_6:
      sub_C3B170((__int64)v11, v4);
      sub_C33870((__int64)a1, (__int64)v11);
      sub_C338F0((__int64)v11);
      return 0;
    }
LABEL_11:
    sub_C36070((__int64)a1, 0, 0, 0);
  }
  return 1;
}
