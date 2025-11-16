// Function: sub_169CC60
// Address: 0x169cc60
//
__int64 __fastcall sub_169CC60(_BYTE *a1)
{
  unsigned int v1; // r8d

  v1 = 0;
  if ( (a1[18] & 7) == 1 )
    LOBYTE(v1) = *(_DWORD *)(*(_QWORD *)a1 + 4LL) <= 0x3Fu;
  return v1;
}
