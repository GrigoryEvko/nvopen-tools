// Function: sub_1E71D10
// Address: 0x1e71d10
//
__int64 __fastcall sub_1E71D10(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax

  result = sub_1E70870(a1, a2, a3, a4, a5);
  if ( *(_BYTE *)(a1 + 2568) )
  {
    result = *(_QWORD *)(a1 + 2240);
    *(_QWORD *)(a1 + 3352) = result;
  }
  return result;
}
