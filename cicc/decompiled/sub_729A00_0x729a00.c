// Function: sub_729A00
// Address: 0x729a00
//
__int64 __fastcall sub_729A00(__int64 a1, int a2)
{
  __int64 result; // rax

  result = qword_4F07B08;
  *(_DWORD *)(a1 + 28) = a2;
  qword_4F07B28 = 0;
  dword_4F07B30 = 0;
  qword_4F07B38 = 0;
  qword_4F07B20 = 0;
  *(_DWORD *)(result + 12) = a2;
  return result;
}
