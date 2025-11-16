// Function: sub_397C300
// Address: 0x397c300
//
__int64 __fastcall sub_397C300(__int64 a1, int a2)
{
  __int64 v3; // rax

  if ( a2 == 255 )
    return 0;
  if ( (a2 & 7) == 3 )
    return 4;
  if ( (a2 & 4) != 0 )
    return 8;
  if ( (a2 & 7) != 0 )
    return 2;
  v3 = sub_1E0A0C0(*(_QWORD *)(a1 + 264));
  return sub_15A9520(v3, 0);
}
