// Function: sub_1346840
// Address: 0x1346840
//
__int64 __fastcall sub_1346840(unsigned __int64 a1)
{
  unsigned int v1; // eax
  int v2; // edx

  v2 = qword_4F96C30;
  LOBYTE(v1) = qword_4F96C40 <= a1;
  LOBYTE(v2) = qword_4F96C30 > a1;
  return v2 & v1;
}
