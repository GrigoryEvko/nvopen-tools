// Function: sub_824E10
// Address: 0x824e10
//
__int64 __fastcall sub_824E10(__int64 a1)
{
  __int64 v1; // rax

  if ( **(_BYTE **)(a1 + 32) != 2 )
    sub_721090();
  v1 = sub_823970(64);
  if ( v1 )
  {
    *(_BYTE *)v1 = 2;
    *(_QWORD *)(v1 + 8) = 0;
    *(_QWORD *)(v1 + 16) = 0;
    *(_QWORD *)(v1 + 24) = 0;
    *(_QWORD *)(v1 + 32) = 0;
    *(_QWORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = 0;
    *(_QWORD *)(v1 + 56) = 0;
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 32) + 32LL) = v1;
  return sub_824DF0((_BYTE *)v1);
}
