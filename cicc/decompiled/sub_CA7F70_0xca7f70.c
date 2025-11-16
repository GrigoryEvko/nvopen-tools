// Function: sub_CA7F70
// Address: 0xca7f70
//
__int64 __fastcall sub_CA7F70(__int64 a1, unsigned int a2)
{
  *(_DWORD *)(a1 + 60) += a2;
  *(_QWORD *)(a1 + 40) += a2;
  return a2;
}
