// Function: sub_3707C20
// Address: 0x3707c20
//
unsigned __int64 __fastcall sub_3707C20(__int64 a1, int a2)
{
  unsigned int v3; // [rsp+1Bh] [rbp-5h]
  unsigned __int8 v4; // [rsp+1Fh] [rbp-1h]

  v3 = sub_3707C10(a1);
  if ( a2 + 1 == v3 )
  {
    v4 = 0;
  }
  else
  {
    v3 = a2 + 1;
    v4 = 1;
  }
  return ((unsigned __int64)v4 << 32) | v3;
}
