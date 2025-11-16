// Function: sub_103D140
// Address: 0x103d140
//
void __fastcall sub_103D140(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r9
  _QWORD v4[4]; // [rsp+0h] [rbp-20h] BYREF

  v2 = *(_QWORD *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 16);
  v4[0] = off_49E5A18;
  v4[1] = a1;
  if ( v2 )
    v3 = *(_QWORD *)(**(_QWORD **)(v2 + 32) + 72LL);
  sub_A68C30(v3, a2, (__int64)v4, 0, 0);
  v4[0] = off_49E5A18;
  nullsub_35();
}
