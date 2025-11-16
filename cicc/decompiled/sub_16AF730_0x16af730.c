// Function: sub_16AF730
// Address: 0x16af730
//
__int64 __fastcall sub_16AF730(unsigned __int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rdx
  char v3; // cl
  unsigned int v5; // [rsp+Ch] [rbp-4h] BYREF

  v2 = a2;
  if ( a2 > 0xFFFFFFFF )
  {
    v3 = 0;
    do
    {
      v2 >>= 1;
      ++v3;
    }
    while ( v2 > 0xFFFFFFFF );
    a1 >>= v3;
  }
  sub_16AF710(&v5, a1, v2);
  return v5;
}
