// Function: sub_2304810
// Address: 0x2304810
//
_QWORD *__fastcall sub_2304810(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  _QWORD *v5; // rax
  _QWORD *v6; // rbx
  __int64 v7; // rax
  _QWORD v9[2]; // [rsp+0h] [rbp-60h] BYREF
  __int64 v10; // [rsp+10h] [rbp-50h]
  __int64 v11; // [rsp+20h] [rbp-40h] BYREF
  __int64 v12; // [rsp+28h] [rbp-38h]
  __int64 v13; // [rsp+30h] [rbp-30h]

  sub_22C1770(v9, a2 + 8, a3, a4);
  v11 = v9[0];
  v12 = v9[1];
  v4 = v10;
  v10 = 0;
  v13 = v4;
  v5 = (_QWORD *)sub_22077B0(0x20u);
  v6 = v5;
  if ( v5 )
  {
    *v5 = &unk_4A0AF70;
    v5[1] = v11;
    v5[2] = v12;
    v7 = v13;
    v13 = 0;
    v6[3] = v7;
  }
  sub_22C31B0((__int64)&v11);
  *a1 = v6;
  sub_22C31B0((__int64)v9);
  return a1;
}
