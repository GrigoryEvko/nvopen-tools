// Function: sub_8DCBF0
// Address: 0x8dcbf0
//
__int64 __fastcall sub_8DCBF0(__int64 a1, __int64 a2, int a3, int a4)
{
  unsigned int v5; // edx
  unsigned int v6; // eax
  unsigned int v7; // eax

  v5 = 6451;
  if ( !a3 )
    v5 = a4 == 0 ? 6691 : 6179;
  dword_4F60570 = a3;
  v6 = v5;
  qword_4F60580 = a2;
  qword_4F60578 = 0;
  dword_4F6056C = 0;
  dword_4F60568 = 0;
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
  return sub_8D9600(a1, (__int64 (__fastcall *)(__int64, unsigned int *))sub_8DB8B0, v5);
}
