// Function: sub_B14070
// Address: 0xb14070
//
__int64 __fastcall sub_B14070(__int64 a1)
{
  unsigned __int8 *v1; // rax
  unsigned int v2; // r8d

  v1 = sub_B13320(a1);
  v2 = 1;
  if ( v1 )
    LOBYTE(v2) = (unsigned int)*v1 - 12 <= 1;
  return v2;
}
