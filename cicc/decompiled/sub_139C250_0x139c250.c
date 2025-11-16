// Function: sub_139C250
// Address: 0x139c250
//
__int64 __fastcall sub_139C250(__int64 a1, __int64 a2)
{
  __int64 v2; // rax

  v2 = sub_1648700(a2);
  if ( *(_BYTE *)(v2 + 16) == 25 && !*(_BYTE *)(a1 + 32) )
    return 0;
  if ( v2 == *(_QWORD *)(a1 + 16) && !*(_BYTE *)(a1 + 33) || (unsigned __int8)sub_139BE50((_QWORD *)a1, v2) )
    return 0;
  *(_BYTE *)(a1 + 34) = 1;
  return 1;
}
