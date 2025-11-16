// Function: sub_302DB60
// Address: 0x302db60
//
__int64 __fastcall sub_302DB60(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  __int64 v3; // rax
  __int64 v4; // r12

  if ( !*(_BYTE *)(a1 + 1096) )
  {
    sub_302CFC0(a1, a2);
    *(_BYTE *)(a1 + 1096) = 1;
  }
  v2 = sub_31E7BD0(a1, a2);
  sub_307A780(a2);
  v3 = *(_QWORD *)(a1 + 224);
  v4 = *(_QWORD *)(v3 + 16);
  if ( *(_BYTE *)(a1 + 782) )
    sub_301FB00(*(_QWORD *)(v3 + 16));
  sub_301F800(v4);
  return v2;
}
