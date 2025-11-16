// Function: sub_1455FA0
// Address: 0x1455fa0
//
__int64 __fastcall sub_1455FA0(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 16);
  if ( result != -8 && result != 0 && result != -16 )
    return sub_1649B30(a1);
  return result;
}
