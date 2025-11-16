// Function: sub_3225810
// Address: 0x3225810
//
__int64 __fastcall sub_3225810(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rsi

  v1 = sub_31DA6B0(*(_QWORD *)(a1 + 8));
  v2 = *(_QWORD *)(v1 + 296);
  if ( !*(_BYTE *)(a1 + 3692) )
    v2 = *(_QWORD *)(v1 + 288);
  return sub_3225490(a1, v2);
}
