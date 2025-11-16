// Function: sub_23379D0
// Address: 0x23379d0
//
_QWORD *__fastcall sub_23379D0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  _QWORD *v5; // rax
  _QWORD *v6; // rbx
  __int64 v7; // rax
  __int64 v9[2]; // [rsp+0h] [rbp-60h] BYREF
  char v10; // [rsp+10h] [rbp-50h]
  __int64 v11; // [rsp+20h] [rbp-40h] BYREF
  __int64 v12; // [rsp+28h] [rbp-38h]
  char v13; // [rsp+30h] [rbp-30h]

  sub_22D0350((__int64)v9, (__int64 *)(a2 + 8), a3, a4);
  v4 = v9[0];
  v9[0] = 0;
  v11 = v4;
  v12 = v9[1];
  v13 = v10;
  v5 = (_QWORD *)sub_22077B0(0x20u);
  v6 = v5;
  if ( v5 )
  {
    *v5 = &unk_4A0AB60;
    v7 = v11;
    v11 = 0;
    v6[1] = v7;
    v6[2] = v12;
    *((_BYTE *)v6 + 24) = v13;
  }
  sub_2337550(&v11);
  *a1 = v6;
  sub_2337550(v9);
  return a1;
}
