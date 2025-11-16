// Function: sub_321F9B0
// Address: 0x321f9b0
//
__int64 __fastcall sub_321F9B0(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // r12
  __int64 v3; // rax

  v1 = 0;
  if ( *(_BYTE *)(a1 + 3770) )
  {
    sub_321F870(a1);
    v1 = *(_QWORD *)(sub_31DA6B0(*(_QWORD *)(a1 + 8)) + 304);
  }
  v2 = a1 + 3080;
  if ( *(_BYTE *)(a1 + 3769) )
    v2 = a1 + 3776;
  v3 = sub_31DA6B0(*(_QWORD *)(a1 + 8));
  return sub_3245030(v2, *(_QWORD *)(v3 + 136), v1, 1);
}
