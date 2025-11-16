// Function: sub_71CE00
// Address: 0x71ce00
//
__int64 __fastcall sub_71CE00(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rcx

  while ( *(_BYTE *)(a1 + 140) == 12 )
    a1 = *(_QWORD *)(a1 + 160);
  for ( result = *(_QWORD *)(a1 + 168); *(_BYTE *)(a2 + 140) == 12; a2 = *(_QWORD *)(a2 + 160) )
    ;
  v3 = *(_QWORD *)(a2 + 168);
  *(_QWORD *)(result + 40) = *(_QWORD *)(v3 + 40);
  *(_BYTE *)(result + 21) = *(_BYTE *)(v3 + 21) & 1 | *(_BYTE *)(result + 21) & 0xFE;
  *(_BYTE *)(result + 18) = *(_BYTE *)(v3 + 18) & 0x7F | *(_BYTE *)(result + 18) & 0x80;
  *(_BYTE *)(result + 17) = *(_BYTE *)(v3 + 17) & 0x70 | *(_BYTE *)(result + 17) & 0x8F;
  return result;
}
