// Function: sub_622A10
// Address: 0x622a10
//
__int64 __fastcall sub_622A10(__int64 a1, int a2, int a3)
{
  __int64 i; // r14
  unsigned int v4; // r15d
  int v7; // [rsp+14h] [rbp-3Ch] BYREF
  _QWORD v8[7]; // [rsp+18h] [rbp-38h] BYREF

  for ( i = 0; i != 13; ++i )
  {
    v4 = i;
    sub_622920((unsigned int)i, v8, &v7);
    if ( v8[0] == a1 && v7 == a2 && byte_4B6DF90[i] == a3 )
      break;
    ++v4;
  }
  return v4;
}
