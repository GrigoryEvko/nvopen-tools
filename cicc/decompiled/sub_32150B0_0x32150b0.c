// Function: sub_32150B0
// Address: 0x32150b0
//
unsigned __int64 __fastcall sub_32150B0(unsigned __int64 a1)
{
  __int64 v1; // rbx

  v1 = 0x201000000000001LL;
  do
  {
    if ( (unsigned __int16)(*(_WORD *)(a1 + 28) - 17) <= 0x39u
      && _bittest64(&v1, (unsigned int)*(unsigned __int16 *)(a1 + 28) - 17) )
    {
      break;
    }
    a1 = sub_3214EE0(a1);
  }
  while ( a1 );
  return a1;
}
