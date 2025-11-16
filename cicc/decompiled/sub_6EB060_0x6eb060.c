// Function: sub_6EB060
// Address: 0x6eb060
//
__int64 __fastcall sub_6EB060(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v6[7]; // [rsp+8h] [rbp-38h] BYREF

  v4 = a2;
  v6[0] = 1;
  while ( (unsigned int)sub_8D3410(v4) )
  {
    sub_6E32E0(v4, v6);
    v4 = sub_8D4050(v4);
  }
  return sub_6EAFD0(a1, a2, v4, a3, v6[0]);
}
