// Function: sub_2EBEAA0
// Address: 0x2ebeaa0
//
__int64 __fastcall sub_2EBEAA0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx

  result = *(_QWORD *)(a1 + 488);
  v2 = *(_QWORD *)(a1 + 496);
  for ( *(_DWORD *)(a1 + 64) = 0; v2 != result; result += 8 )
    *(_DWORD *)(result + 4) = 0;
  return result;
}
