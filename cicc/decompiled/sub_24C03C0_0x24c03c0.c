// Function: sub_24C03C0
// Address: 0x24c03c0
//
char *__fastcall sub_24C03C0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD v4[4]; // [rsp+0h] [rbp-60h] BYREF
  __int16 v5; // [rsp+20h] [rbp-40h]
  _QWORD v6[4]; // [rsp+30h] [rbp-30h] BYREF
  __int16 v7; // [rsp+50h] [rbp-10h]

  v4[2] = a1 + 80;
  v5 = 1029;
  v4[0] = a2;
  v4[1] = a3;
  v6[0] = v4;
  v6[2] = "!C";
  v7 = 770;
  return sub_C94C70(a1 + 352, (__int64)v6);
}
