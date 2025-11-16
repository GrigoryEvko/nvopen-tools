// Function: sub_12FB880
// Address: 0x12fb880
//
__int16 __fastcall sub_12FB880(unsigned __int8 *a1)
{
  unsigned __int64 v1; // rax

  v1 = (*((_QWORD *)a1 + 2) >> 54) | ((unsigned __int64)*a1 << 15);
  BYTE1(v1) |= 0x7Eu;
  return v1;
}
