// Function: sub_1302A70
// Address: 0x1302a70
//
__int64 __fastcall sub_1302A70(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax

  sub_1317800(a2, 0);
  sub_13177F0(a3, 0);
  *(_QWORD *)(a1 + 144) = a3;
  result = sub_1317080(a2, 0);
  if ( !(_DWORD)result )
    return sub_1315160(a1, a2, 0, 1);
  return result;
}
