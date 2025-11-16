// Function: sub_AA7C30
// Address: 0xaa7c30
//
__int64 __fastcall sub_AA7C30(
        __int64 a1,
        __int64 a2,
        char a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 result; // rax
  __int64 v12; // r12
  __int64 v13; // rdi
  __int64 v14; // rdi
  _QWORD *v15; // rax
  _QWORD *v16; // rax
  __int64 v17; // r8
  _QWORD *v18; // rax
  __int64 v19; // rdi
  _QWORD *v20; // rax
  _QWORD *v21; // rax
  __int64 v22; // [rsp+0h] [rbp-60h]
  __int64 v24; // [rsp+20h] [rbp-40h]
  _QWORD *v25; // [rsp+20h] [rbp-40h]
  __int64 v26; // [rsp+20h] [rbp-40h]

  v22 = a4 + 48;
  result = sub_AA6160(a1, a2);
  v12 = result;
  if ( result )
  {
    if ( a2 == a1 + 48 )
    {
      result = sub_AA6260(a1);
      if ( BYTE1(a8) )
        goto LABEL_5;
      goto LABEL_19;
    }
    result = sub_B141E0(result);
  }
  if ( BYTE1(a8) )
    goto LABEL_5;
LABEL_19:
  result = sub_AA6160(a4, a7);
  if ( result )
  {
    v17 = sub_AA6160(a4, a7);
    if ( v22 == a7 )
    {
      if ( a2 == a1 + 48 )
      {
        v26 = v17;
        v21 = sub_AA7AD0(a1, a2);
        sub_B14410(v21, v26, 1);
        sub_B14200(v26);
        result = sub_AA6260(a4);
      }
      else
      {
        v19 = a2 - 24;
        if ( !a2 )
          v19 = 0;
        result = sub_B44050(v19, a4, v22, a8, 1);
      }
    }
    else
    {
      v24 = v17;
      v18 = sub_AA7AD0(a1, a2);
      result = sub_B14410(v18, v24, 1);
    }
  }
LABEL_5:
  if ( !(_BYTE)a6 )
  {
    v13 = a5 - 24;
    if ( !a5 )
      v13 = 0;
    result = sub_B44020(v13);
    if ( (_BYTE)result )
    {
      if ( v22 == a7 )
      {
        v25 = sub_AA7AD0(a4, v22);
        v20 = sub_AA7AD0(a4, a5);
        result = sub_B14410(v25, v20, 1);
      }
      else
      {
        v14 = a7 - 24;
        if ( !a7 )
          v14 = 0;
        result = sub_B44050(v14, a4, a5, a6, 1);
      }
    }
  }
  if ( v12 )
  {
    if ( a3 )
    {
      v16 = sub_AA7AD0(a1, a2);
      sub_B14410(v16, v12, 0);
    }
    else
    {
      v15 = sub_AA7AD0(a1, a5);
      sub_B14410(v15, v12, 1);
    }
    return sub_B14200(v12);
  }
  return result;
}
