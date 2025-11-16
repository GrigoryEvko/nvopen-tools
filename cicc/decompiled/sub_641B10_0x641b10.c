// Function: sub_641B10
// Address: 0x641b10
//
__int64 __fastcall sub_641B10(__int64 *a1)
{
  __int64 result; // rax
  __int64 v2; // r8

  result = *a1;
  if ( *(_BYTE *)(*a1 + 80) == 11 )
  {
    v2 = *(_QWORD *)(result + 88);
    result = *(_QWORD *)(v2 + 152);
    if ( *(_BYTE *)(result + 140) == 7 )
    {
      result = *(_BYTE *)(*(_QWORD *)(a1[36] + 168) + 17LL) & 0x70;
      if ( (_BYTE)result == 32 )
        return sub_814390(v2);
    }
  }
  return result;
}
