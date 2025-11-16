// Function: sub_325E6A0
// Address: 0x325e6a0
//
__int64 __fastcall sub_325E6A0(__int64 a1, unsigned __int16 *a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d

  v2 = *a2;
  v3 = 0;
  if ( (_WORD)v2 )
    LOBYTE(v3) = *(_QWORD *)(a1 + 8 * v2 + 112) != 0;
  return v3;
}
