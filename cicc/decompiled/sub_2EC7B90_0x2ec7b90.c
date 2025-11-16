// Function: sub_2EC7B90
// Address: 0x2ec7b90
//
__int64 __fastcall sub_2EC7B90(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax

  result = sub_2EC65D0(a1, a2, a3, a4, a5);
  if ( *(_BYTE *)(a1 + 4016) )
  {
    result = *(_QWORD *)(a1 + 3504);
    *(_QWORD *)(a1 + 5440) = result;
  }
  return result;
}
