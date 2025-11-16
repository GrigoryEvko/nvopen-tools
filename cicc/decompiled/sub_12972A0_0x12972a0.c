// Function: sub_12972A0
// Address: 0x12972a0
//
__int64 __fastcall sub_12972A0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rbx

  if ( *(char *)(a2 + 196) < 0 )
    sub_15606E0(a1, 37);
  for ( result = *(_QWORD *)(a2 + 152); *(_BYTE *)(result + 140) == 12; result = *(_QWORD *)(result + 160) )
    ;
  v3 = *(_QWORD *)(result + 168);
  if ( v3 )
  {
    result = *(unsigned __int8 *)(v3 + 20);
    if ( (result & 8) == 0 )
    {
      if ( (result & 1) == 0 )
        return result;
      return sub_15606E0(a1, 29);
    }
    sub_15606E0(a1, 36);
    result = *(unsigned __int8 *)(v3 + 20);
    if ( (result & 1) != 0 )
      return sub_15606E0(a1, 29);
  }
  return result;
}
