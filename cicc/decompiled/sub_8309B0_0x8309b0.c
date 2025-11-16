// Function: sub_8309B0
// Address: 0x8309b0
//
__int64 __fastcall sub_8309B0(_QWORD *a1)
{
  __int64 i; // rax
  __int64 v2; // rax

  for ( i = *(_QWORD *)(a1[4] + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v2 = **(_QWORD **)(i + 168);
  if ( v2 && (*(_BYTE *)(v2 + 35) & 1) != 0 )
    return a1[5];
  else
    return a1[8];
}
