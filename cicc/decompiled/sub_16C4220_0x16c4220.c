// Function: sub_16C4220
// Address: 0x16c4220
//
char *__fastcall sub_16C4220(unsigned __int8 *a1, unsigned __int64 a2, int a3)
{
  _QWORD v5[2]; // [rsp+0h] [rbp-90h] BYREF
  char *v6; // [rsp+10h] [rbp-80h]
  unsigned __int64 v7; // [rsp+18h] [rbp-78h]
  _QWORD v8[12]; // [rsp+30h] [rbp-60h] BYREF

  sub_16C36E0((__int64)v5, a1, a2, a3);
  sub_16C3680(v8, (__int64)a1, a2);
  if ( !sub_16C36A0(v5, v8) && (v7 > 2 && sub_16C36C0(*v6, a3) && *v6 == v6[1] || !a3 && v7 && v6[v7 - 1] == 58) )
    return v6;
  else
    return 0;
}
