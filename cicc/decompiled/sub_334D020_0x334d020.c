// Function: sub_334D020
// Address: 0x334d020
//
__int64 __fastcall sub_334D020(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x2B0u);
  v2 = v1;
  if ( v1 )
  {
    sub_335DCC0(v1, *(_QWORD *)(a1 + 40));
    *(_QWORD *)(v2 + 632) = 0;
    *(_QWORD *)v2 = off_4A361B0;
    *(_QWORD *)(v2 + 640) = 0;
    *(_QWORD *)(v2 + 648) = 0;
    *(_QWORD *)(v2 + 656) = 0;
    *(_QWORD *)(v2 + 664) = 0;
    *(_QWORD *)(v2 + 672) = 0;
    *(_DWORD *)(v2 + 680) = 0;
  }
  return v2;
}
