// Function: sub_C3B200
// Address: 0xc3b200
//
__int64 __fastcall sub_C3B200(__int64 *a1, __int64 *a2, __int64 *a3, char a4)
{
  float v4; // xmm0_4
  __m128i v5; // xmm0
  float v7; // xmm0_4
  float v8; // [rsp+8h] [rbp-48h]
  float v9; // [rsp+8h] [rbp-48h]
  float v10; // [rsp+8h] [rbp-48h]
  float v11; // [rsp+8h] [rbp-48h]
  float v12; // [rsp+Ch] [rbp-44h]
  float v13; // [rsp+Ch] [rbp-44h]
  float v14; // [rsp+Ch] [rbp-44h]
  float v15; // [rsp+Ch] [rbp-44h]
  char v16; // [rsp+1Fh] [rbp-31h] BYREF
  _BYTE v17[48]; // [rsp+20h] [rbp-30h] BYREF

  if ( a4 == 2 )
  {
    v11 = sub_C3AAA0(a3);
    v15 = sub_C3AAA0(a2);
    *(float *)v5.m128i_i32 = sub_C3AAA0(a1);
    sub_CED4E0(&v16, 1, 1, *(float *)v5.m128i_i32, v15, v11);
    goto LABEL_9;
  }
  if ( a4 > 2 )
  {
    if ( a4 != 3 )
      return 1;
    v9 = sub_C3AAA0(a3);
    v13 = sub_C3AAA0(a2);
    v7 = sub_C3AAA0(a1);
    *(float *)v5.m128i_i32 = sub_CED4F0(&v16, 1, 1, v7, v13, v9);
LABEL_9:
    if ( !(unsigned int)sub_CED540(&v16) )
    {
      v5 = (__m128i)v5.m128i_u32[0];
      goto LABEL_6;
    }
    goto LABEL_11;
  }
  if ( !a4 )
  {
    v10 = sub_C3AAA0(a3);
    v14 = sub_C3AAA0(a2);
    *(float *)v5.m128i_i32 = sub_C3AAA0(a1);
    sub_CED4D0(&v16, 1, 1, *(float *)v5.m128i_i32, v14, v10);
    goto LABEL_9;
  }
  if ( a4 == 1 )
  {
    v8 = sub_C3AAA0(a3);
    v12 = sub_C3AAA0(a2);
    v4 = sub_C3AAA0(a1);
    v5 = (__m128i)COERCE_UNSIGNED_INT(sub_CED4C0(&v16, 1, 1, v4, v12, v8));
    if ( !(unsigned int)sub_CED540(&v16) )
    {
LABEL_6:
      sub_C3B170((__int64)v17, v5);
      sub_C33870((__int64)a1, (__int64)v17);
      sub_C338F0((__int64)v17);
      return 0;
    }
LABEL_11:
    sub_C36070((__int64)a1, 0, 0, 0);
  }
  return 1;
}
