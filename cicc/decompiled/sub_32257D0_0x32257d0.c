// Function: sub_32257D0
// Address: 0x32257d0
//
__int64 __fastcall sub_32257D0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rsi

  v1 = sub_31DA6B0(*(_QWORD *)(a1 + 8));
  v2 = *(_QWORD *)(v1 + 176);
  if ( !*(_BYTE *)(a1 + 3692) )
    v2 = *(_QWORD *)(v1 + 168);
  return sub_3225490(a1, v2);
}
