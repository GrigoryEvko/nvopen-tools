// Function: sub_321F870
// Address: 0x321f870
//
__int64 __fastcall sub_321F870(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // r13
  __int64 v3; // rax

  v1 = a1 + 3776;
  if ( !*(_BYTE *)(a1 + 3769) )
    v1 = a1 + 3080;
  v2 = *(_QWORD *)(v1 + 320);
  v3 = sub_31DA6B0(*(_QWORD *)(a1 + 8));
  return sub_32468F0(v1 + 176, *(_QWORD *)(a1 + 8), *(_QWORD *)(v3 + 304), v2);
}
