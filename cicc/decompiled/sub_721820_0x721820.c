// Function: sub_721820
// Address: 0x721820
//
_BOOL8 __fastcall sub_721820(__int64 a1, _QWORD *a2)
{
  _BOOL8 result; // rax

  result = 0;
  if ( *(_QWORD *)a1 == *a2 && *(_QWORD *)(a1 + 8) == a2[1] )
    return *(_OWORD *)a1 != 0;
  return result;
}
