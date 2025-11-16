// Function: sub_1E69990
// Address: 0x1e69990
//
__int64 __fastcall sub_1E69990(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx

  result = *(_QWORD *)(a1 + 360);
  v2 = *(_QWORD *)(a1 + 368);
  for ( *(_DWORD *)(a1 + 32) = 0; v2 != result; result += 8 )
    *(_DWORD *)(result + 4) = 0;
  return result;
}
