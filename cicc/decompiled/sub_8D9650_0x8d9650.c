// Function: sub_8D9650
// Address: 0x8d9650
//
__int64 __fastcall sub_8D9650(__int64 a1)
{
  unsigned int v1; // edx
  unsigned int v2; // eax

  qword_4F60580 = 0;
  qword_4F60578 = 0;
  dword_4F60570 = 1;
  dword_4F6056C = 0;
  dword_4F60568 = 0;
  v1 = unk_4D04318 == 0 ? 6419 : 6931;
  v2 = v1;
  if ( dword_4D04804 )
  {
    BYTE1(v2) = BYTE1(v1) | 0x20;
    v1 = v2;
  }
  return sub_8D9600(a1, sub_8D1FA0, v1);
}
