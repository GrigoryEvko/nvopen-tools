// Function: sub_398A640
// Address: 0x398a640
//
__int64 __fastcall sub_398A640(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rax

  v1 = a1 + 4520;
  if ( !*(_BYTE *)(a1 + 4513) )
    v1 = a1 + 4040;
  v2 = sub_396DD80(*(_QWORD *)(a1 + 8));
  return sub_39A0350(v1, *(_QWORD *)(v2 + 80));
}
