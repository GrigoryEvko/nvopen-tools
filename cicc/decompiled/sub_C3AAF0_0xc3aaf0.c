// Function: sub_C3AAF0
// Address: 0xc3aaf0
//
__int64 __fastcall sub_C3AAF0(__int64 *a1, __int64 *a2, char a3)
{
  float v3; // xmm0_4
  float v4; // xmm0_4
  float v6; // xmm0_4
  float v7; // [rsp+Ch] [rbp-44h]
  float v8; // [rsp+Ch] [rbp-44h]
  float v9; // [rsp+Ch] [rbp-44h]
  float v10; // [rsp+Ch] [rbp-44h]
  char v11; // [rsp+1Fh] [rbp-31h] BYREF
  _BYTE v12[48]; // [rsp+20h] [rbp-30h] BYREF

  if ( a3 == 2 )
  {
    v10 = sub_C3AAA0(a2);
    v4 = sub_C3AAA0(a1);
    sub_CED470(&v11, 1, 1, v4, v10);
    goto LABEL_9;
  }
  if ( a3 > 2 )
  {
    if ( a3 != 3 )
      return 1;
    v8 = sub_C3AAA0(a2);
    v6 = sub_C3AAA0(a1);
    v4 = sub_CED480(&v11, 1, 1, v6, v8);
LABEL_9:
    if ( !(unsigned int)sub_CED540(&v11) )
      goto LABEL_6;
LABEL_10:
    sub_C36070((__int64)a1, 0, 0, 0);
    return 1;
  }
  if ( !a3 )
  {
    v9 = sub_C3AAA0(a2);
    v4 = sub_C3AAA0(a1);
    sub_CED460(&v11, 1, 1, v4, v9);
    goto LABEL_9;
  }
  if ( a3 == 1 )
  {
    v7 = sub_C3AAA0(a2);
    v3 = sub_C3AAA0(a1);
    v4 = sub_CED450(&v11, 1, 1, v3, v7);
    if ( !(unsigned int)sub_CED540(&v11) )
    {
LABEL_6:
      sub_C3B170(v12, v4);
      sub_C33870((__int64)a1, (__int64)v12);
      sub_C338F0((__int64)v12);
      return 0;
    }
    goto LABEL_10;
  }
  return 1;
}
