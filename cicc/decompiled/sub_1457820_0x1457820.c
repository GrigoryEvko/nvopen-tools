// Function: sub_1457820
// Address: 0x1457820
//
__int64 __fastcall sub_1457820(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int16 v3; // dx

  for ( result = a2; ; result = *(_QWORD *)(result + 32) )
  {
    v3 = *(_WORD *)(result + 24);
    if ( v3 != 2 && v3 != 3 )
      break;
  }
  return result;
}
