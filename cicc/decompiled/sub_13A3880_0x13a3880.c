// Function: sub_13A3880
// Address: 0x13a3880
//
__int64 __fastcall sub_13A3880(unsigned __int64 a1)
{
  __int64 v1; // r12
  __int64 v3; // rax

  v1 = malloc(a1);
  if ( !v1 )
  {
    if ( a1 || (v3 = malloc(1u)) == 0 )
      sub_16BD1C0("Allocation failed");
    else
      return v3;
  }
  return v1;
}
