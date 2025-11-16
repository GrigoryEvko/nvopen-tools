// Function: sub_8DCC70
// Address: 0x8dcc70
//
__int64 __fastcall sub_8DCC70(__int64 a1, __int64 a2, int a3, int a4)
{
  unsigned int v5; // edx
  unsigned int v6; // eax
  unsigned int v7; // eax

  v5 = 4387;
  if ( !a3 )
    v5 = a4 == 0 ? 4643 : 4131;
  v6 = v5;
  qword_4F60560 = a2;
  if ( unk_4D04318 )
  {
    BYTE1(v6) = BYTE1(v5) | 2;
    v5 = v6;
  }
  if ( dword_4D04804 )
  {
    v7 = v5;
    if ( (v5 & 0x100) != 0 )
    {
      BYTE1(v7) = BYTE1(v5) | 0x20;
      v5 = v7;
    }
  }
  return sub_8D9600(a1, sub_8D1880, v5);
}
