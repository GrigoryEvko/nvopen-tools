// Function: sub_D98140
// Address: 0xd98140
//
__int64 __fastcall sub_D98140(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int16 v3; // dx

  for ( result = a2; ; result = *(_QWORD *)(result + 32) )
  {
    v3 = *(_WORD *)(result + 24);
    if ( v3 != 3 && v3 != 4 )
      break;
  }
  return result;
}
