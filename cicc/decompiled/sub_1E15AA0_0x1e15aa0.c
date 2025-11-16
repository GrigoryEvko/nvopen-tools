// Function: sub_1E15AA0
// Address: 0x1e15aa0
//
__int64 __fastcall sub_1E15AA0(__int64 a1, __int64 a2)
{
  __int64 i; // r12

  for ( i = *(_QWORD *)(a2 + 8); a1 + 24 != i; i = *(_QWORD *)(i + 8) )
  {
    if ( (*(_BYTE *)(i + 46) & 4) == 0 )
      break;
  }
  sub_1E14280(a1, a2, (_BYTE *)i);
  return i;
}
