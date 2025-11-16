// Function: sub_398A680
// Address: 0x398a680
//
__int64 __fastcall sub_398A680(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // r13
  __int64 v3; // rax

  v1 = a1 + 4520;
  if ( !*(_BYTE *)(a1 + 4513) )
    v1 = a1 + 4040;
  v2 = *(_QWORD *)(v1 + 248);
  v3 = sub_396DD80(*(_QWORD *)(a1 + 8));
  return sub_39A11F0(v1 + 192, *(_QWORD *)(a1 + 8), *(_QWORD *)(v3 + 280), v2);
}
