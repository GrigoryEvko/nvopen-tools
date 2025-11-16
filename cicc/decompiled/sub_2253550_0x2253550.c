// Function: sub_2253550
// Address: 0x2253550
//
__int64 __fastcall sub_2253550(unsigned __int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rbp
  unsigned __int64 v3; // rbx
  __int64 result; // rax
  void (*v5)(void); // rax
  __int64 v6[4]; // [rsp+8h] [rbp-20h] BYREF

  v2 = a1;
  v3 = a2;
  if ( (unsigned int)sub_39FAC40(a2) != 1 )
    JUMPOUT(0x426734);
  if ( !a1 )
    v2 = 1;
  if ( a2 < 8 )
    v3 = 8;
  while ( 1 )
  {
    if ( !(unsigned int)posix_memalign(v6, v3, v2) )
    {
      result = v6[0];
      if ( v6[0] )
        break;
    }
    v5 = (void (*)(void))sub_22077A0();
    if ( !v5 )
      JUMPOUT(0x426762);
    v5();
  }
  return result;
}
