// Function: sub_C771C0
// Address: 0xc771c0
//
__int64 __fastcall sub_C771C0(__int64 *a1, __int64 a2)
{
  unsigned __int16 v2; // ax
  unsigned int v3; // edx

  v2 = sub_C76FF0(a1, a2);
  v3 = 0;
  if ( HIBYTE(v2) )
  {
    LOBYTE(v3) = v2 ^ 1;
    BYTE1(v3) = 1;
  }
  return v3;
}
