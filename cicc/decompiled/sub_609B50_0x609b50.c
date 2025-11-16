// Function: sub_609B50
// Address: 0x609b50
//
__int64 __fastcall sub_609B50(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rbx
  int v4; // [rsp+4h] [rbp-5Ch] BYREF
  __int64 v5; // [rsp+8h] [rbp-58h] BYREF
  __int64 v6; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v7; // [rsp+18h] [rbp-48h]
  char v8; // [rsp+1Ch] [rbp-44h]
  __int64 v9; // [rsp+20h] [rbp-40h]
  __int64 v10; // [rsp+28h] [rbp-38h]
  __int64 v11; // [rsp+30h] [rbp-30h]
  __int64 v12; // [rsp+38h] [rbp-28h]
  __int64 v13; // [rsp+40h] [rbp-20h]

  v6 = a1;
  v2 = qword_4CF8008;
  v7 = v7 & 0xF8000000 | 1;
  v5 = 0;
  v8 = 0;
  v9 = 0;
  v10 = 0;
  v11 = 0;
  v12 = 0;
  v13 = 0;
  qword_4CF8008 = 0;
  sub_6040F0((__int64)&v6, 0, 0, 0, 0, &v4, &v5, a2, 0, 0);
  qword_4CF8008 = v2;
  return v5;
}
