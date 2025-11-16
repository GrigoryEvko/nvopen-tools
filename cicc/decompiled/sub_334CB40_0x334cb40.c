// Function: sub_334CB40
// Address: 0x334cb40
//
__int64 __fastcall sub_334CB40(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x340u);
  v2 = v1;
  if ( v1 )
  {
    sub_335DCC0(v1, *(_QWORD *)(a1 + 40));
    *(_DWORD *)(v2 + 776) = 0;
    *(_QWORD *)v2 = off_4A36128;
    *(_QWORD *)(v2 + 632) = v2 + 648;
    *(_QWORD *)(v2 + 640) = 0x1000000000LL;
    *(_QWORD *)(v2 + 784) = 0;
    *(_QWORD *)(v2 + 792) = 0;
    *(_QWORD *)(v2 + 800) = 0;
    *(_QWORD *)(v2 + 808) = 0;
    *(_QWORD *)(v2 + 816) = 0;
    *(_QWORD *)(v2 + 824) = 0;
  }
  return v2;
}
