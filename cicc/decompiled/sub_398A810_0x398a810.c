// Function: sub_398A810
// Address: 0x398a810
//
__int64 __fastcall sub_398A810(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // r12
  __int64 v3; // rax

  v1 = 0;
  if ( *(_BYTE *)(a1 + 4514) )
  {
    sub_398A680(a1);
    v1 = *(_QWORD *)(sub_396DD80(*(_QWORD *)(a1 + 8)) + 280);
  }
  v2 = a1 + 4040;
  if ( *(_BYTE *)(a1 + 4513) )
    v2 = a1 + 4520;
  v3 = sub_396DD80(*(_QWORD *)(a1 + 8));
  return sub_39A0370(v2, *(_QWORD *)(v3 + 136), v1, 1);
}
