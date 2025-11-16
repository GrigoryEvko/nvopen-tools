// Function: sub_AA7E90
// Address: 0xaa7e90
//
__int64 __fastcall sub_AA7E90(
        __int64 a1,
        __int64 a2,
        char a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v10; // rax
  __int64 v11; // r11
  __int64 v12; // r9
  __int64 result; // rax
  __int64 v14; // r8
  char v15; // al
  __int64 v16; // rdi
  __int64 v17; // r9
  _QWORD *v18; // rax
  _QWORD *v19; // rax
  char v20; // al
  __int64 v21; // [rsp-10h] [rbp-80h]
  __int64 v22; // [rsp+8h] [rbp-68h]
  __int64 v23; // [rsp+10h] [rbp-60h]
  __int64 v24; // [rsp+10h] [rbp-60h]
  __int64 v25; // [rsp+18h] [rbp-58h]
  __int64 v26; // [rsp+18h] [rbp-58h]
  __int64 v27; // [rsp+18h] [rbp-58h]
  __int64 v28; // [rsp+18h] [rbp-58h]
  char v29; // [rsp+20h] [rbp-50h]
  __int64 v30; // [rsp+20h] [rbp-50h]
  __int64 v32; // [rsp+28h] [rbp-48h]
  __int64 v33; // [rsp+28h] [rbp-48h]

  v29 = a6;
  v10 = sub_AA60B0(a1);
  v11 = a1 + 48;
  v12 = a6;
  if ( a1 + 48 != a2 )
    return sub_AA7C30(a1, a2, a3, a4, a5, a6, a7, a8);
  if ( a3 == 1 )
    return sub_AA7C30(a1, a2, a3, a4, a5, a6, a7, a8);
  v14 = v10;
  if ( !v10 )
    return sub_AA7C30(a1, a2, a3, a4, a5, a6, a7, a8);
  if ( v29 )
  {
    v30 = 0;
LABEL_10:
    v16 = a5 - 24;
    if ( !a5 )
      v16 = 0;
    goto LABEL_12;
  }
  if ( a5 )
  {
    v25 = v10;
    v15 = sub_B44020(a5 - 24);
    v16 = a5 - 24;
    v14 = v25;
    v30 = 0;
    v11 = a1 + 48;
    v12 = a6;
    if ( v15 )
    {
LABEL_8:
      v23 = v11;
      v26 = v14;
      v32 = v12;
      v30 = sub_AA6160(a4, a5);
      sub_B141E0(v30);
      v12 = v32;
      v14 = v26;
      v11 = v23;
      goto LABEL_10;
    }
  }
  else
  {
    v28 = a6;
    v33 = v10;
    v20 = sub_B44020(0);
    v14 = v33;
    v11 = a1 + 48;
    v12 = v28;
    if ( v20 )
      goto LABEL_8;
    v30 = 0;
    v16 = 0;
  }
LABEL_12:
  v24 = v12;
  v27 = v14;
  v22 = v11;
  if ( (unsigned __int8)sub_B44020(v16) )
  {
    sub_B44050(v16, a1, v22, 0, 1);
  }
  else
  {
    v19 = sub_AA4580(a4, v16);
    sub_B14410(v19, v27, 0);
    sub_B14200(v27);
  }
  sub_AA6260(a1);
  v17 = v24;
  LOBYTE(v17) = 1;
  sub_AA7C30(a1, a2, a3, a4, a5, v17, a7, a8);
  result = v21;
  if ( v30 )
  {
    v18 = sub_AA7AD0(a4, a7);
    sub_B14410(v18, v30, 1);
    return sub_B14200(v30);
  }
  return result;
}
