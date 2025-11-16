// Function: sub_C80FE0
// Address: 0xc80fe0
//
char *__fastcall sub_C80FE0(unsigned __int8 *a1, unsigned __int64 a2, unsigned int a3)
{
  _QWORD v5[2]; // [rsp+0h] [rbp-90h] BYREF
  char *v6; // [rsp+10h] [rbp-80h]
  unsigned __int64 v7; // [rsp+18h] [rbp-78h]
  _QWORD v8[12]; // [rsp+30h] [rbp-60h] BYREF

  sub_C80240((__int64)v5, a1, a2, a3);
  sub_C801E0((__int64)v8, (__int64)a1, a2);
  if ( !sub_C80200(v5, v8) && (v7 > 2 && sub_C80220(*v6, a3) && *v6 == v6[1] || a3 > 1 && v7 && v6[v7 - 1] == 58) )
    return v6;
  else
    return 0;
}
