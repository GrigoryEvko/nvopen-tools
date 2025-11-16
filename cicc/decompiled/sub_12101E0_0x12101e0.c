// Function: sub_12101E0
// Address: 0x12101e0
//
__int64 __fastcall sub_12101E0(_BYTE *a1, size_t a2)
{
  int v3; // edx
  unsigned int v4; // eax

  if ( a2 <= 8 || *(_QWORD *)a1 != 0x6762642E6D766C6CLL || a1[8] != 46 )
    return 0;
  v3 = sub_B60C50(a1, a2);
  v4 = v3 - 68;
  LOBYTE(v4) = (unsigned int)(v3 - 68) <= 1;
  LOBYTE(v3) = v3 == 71;
  return v3 | v4;
}
