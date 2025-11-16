// Function: sub_28F6270
// Address: 0x28f6270
//
__int64 __fastcall sub_28F6270(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax

  result = sub_28EB620(30, a3);
  if ( !result && *(_DWORD *)(a3 + 8) != 1 )
    return sub_28F5920(a1, a2, (_QWORD *)a3);
  return result;
}
