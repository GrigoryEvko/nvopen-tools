// Function: sub_D4B130
// Address: 0xd4b130
//
__int64 __fastcall sub_D4B130(__int64 a1)
{
  __int64 v1; // rax
  __int64 v3; // [rsp+8h] [rbp-28h] BYREF
  char v4[8]; // [rsp+10h] [rbp-20h] BYREF
  int v5; // [rsp+18h] [rbp-18h]
  int v6; // [rsp+28h] [rbp-8h]

  v1 = sub_D47840(a1);
  v3 = v1;
  if ( v1 && sub_AA5B10(v1) && (sub_B1C7C0((__int64)v4, &v3), v5 != v6) && v6 == v5 + 1 )
    return v3;
  else
    return 0;
}
