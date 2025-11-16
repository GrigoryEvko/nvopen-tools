// Function: sub_C77420
// Address: 0xc77420
//
__int64 __fastcall sub_C77420(__int64 a1, __int64 a2)
{
  unsigned __int16 v2; // ax
  unsigned int v3; // edx

  v2 = sub_C771E0(a2, a1);
  v3 = 0;
  if ( HIBYTE(v2) )
  {
    LOBYTE(v3) = v2 ^ 1;
    BYTE1(v3) = 1;
  }
  return v3;
}
