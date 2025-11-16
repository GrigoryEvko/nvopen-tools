// Function: sub_22077B0
// Address: 0x22077b0
//
__int64 __fastcall sub_22077B0(unsigned __int64 a1)
{
  unsigned __int64 v1; // rbx
  __int64 result; // rax
  void (*v3)(void); // rax

  v1 = 1;
  if ( a1 )
    v1 = a1;
  while ( 1 )
  {
    result = malloc(v1);
    if ( result )
      break;
    v3 = (void (*)(void))sub_22077A0();
    if ( !v3 )
      JUMPOUT(0x424F5B);
    v3();
  }
  return result;
}
