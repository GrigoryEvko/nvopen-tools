// Function: sub_B4DC00
// Address: 0xb4dc00
//
__int64 __fastcall sub_B4DC00(__int64 a1, unsigned __int64 a2)
{
  int v2; // eax
  __int64 result; // rax
  unsigned int v4; // edx

  v2 = *(unsigned __int8 *)(a1 + 8);
  if ( (_BYTE)v2 == 15 )
  {
    if ( a2 >= *(unsigned int *)(a1 + 12) )
      return 0;
    else
      return *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8 * a2);
  }
  else if ( (_BYTE)v2 == 16 )
  {
    return *(_QWORD *)(a1 + 24);
  }
  else
  {
    v4 = v2 - 17;
    result = 0;
    if ( v4 <= 1 )
      return *(_QWORD *)(a1 + 24);
  }
  return result;
}
