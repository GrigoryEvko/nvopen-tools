// Function: sub_B495C0
// Address: 0xb495c0
//
__int64 __fastcall sub_B495C0(__int64 a1, int a2)
{
  __int64 v2; // rbp
  __int64 v3; // rcx
  _QWORD v5[2]; // [rsp-10h] [rbp-10h] BYREF

  v3 = *(_QWORD *)(a1 - 32);
  if ( *(_BYTE *)v3 )
    return 0;
  v5[1] = v2;
  v5[0] = *(_QWORD *)(v3 + 120);
  return sub_A747F0(v5, -1, a2);
}
