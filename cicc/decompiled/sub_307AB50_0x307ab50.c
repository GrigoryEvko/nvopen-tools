// Function: sub_307AB50
// Address: 0x307ab50
//
__int64 __fastcall sub_307AB50(__int16 a1, __int64 a2, int a3)
{
  int v3; // eax
  int v4; // eax

  LOBYTE(v3) = a1 != 138;
  LOBYTE(a3) = a1 != 127;
  v4 = a3 & v3;
  LOBYTE(a3) = a1 != 47;
  return a3 & v4 ^ 1u;
}
