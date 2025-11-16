// Function: sub_64F530
// Address: 0x64f530
//
__int64 __fastcall sub_64F530(__int64 a1)
{
  __int64 i; // rax
  __int64 v2; // rbx
  __int64 result; // rax

  for ( i = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v2 = *(_QWORD *)(i + 168);
  result = sub_854840(3, a1, 0, 0);
  if ( result )
  {
    *(_BYTE *)(v2 + 16) |= 0x10u;
    result = sub_854000(result);
  }
  if ( (*(_BYTE *)(v2 + 16) & 2) == 0 )
  {
    result = sub_854840(4, a1, 0, 0);
    if ( result )
    {
      *(_WORD *)(v2 + 22) = *(_WORD *)(result + 96);
      return sub_854000(result);
    }
  }
  return result;
}
