// Function: sub_2054600
// Address: 0x2054600
//
__int64 __fastcall sub_2054600(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = *(_QWORD *)(a2 + 8);
  if ( result == *(_QWORD *)(*(_QWORD *)(a1 + 712) + 8LL) + 320LL )
    return 0;
  return result;
}
