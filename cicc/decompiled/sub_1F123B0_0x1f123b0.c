// Function: sub_1F123B0
// Address: 0x1f123b0
//
__int64 __fastcall sub_1F123B0(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax

  result = (*a2 >> 13) + ((*a2 >> 12) & 1LL);
  if ( !result )
    result = 1;
  *(_QWORD *)(a1 + 456) = result;
  return result;
}
