// Function: sub_398BAB0
// Address: 0x398bab0
//
__int64 __fastcall sub_398BAB0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rax

  if ( *(_BYTE *)(a1 + 4514) )
    sub_398BA70(a1);
  v1 = *(_QWORD *)(sub_396DD80(*(_QWORD *)(a1 + 8)) + 272);
  v2 = sub_396DD80(*(_QWORD *)(a1 + 8));
  return sub_39A0370(a1 + 4040, *(_QWORD *)(v2 + 248), v1, 0);
}
