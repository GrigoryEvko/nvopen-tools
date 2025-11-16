// Function: sub_32209A0
// Address: 0x32209a0
//
__int64 __fastcall sub_32209A0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rax

  if ( *(_BYTE *)(a1 + 3770) )
    sub_3220960(a1);
  v1 = *(_QWORD *)(sub_31DA6B0(*(_QWORD *)(a1 + 8)) + 280);
  v2 = sub_31DA6B0(*(_QWORD *)(a1 + 8));
  return sub_3245030(a1 + 3080, *(_QWORD *)(v2 + 256), v1, 0);
}
