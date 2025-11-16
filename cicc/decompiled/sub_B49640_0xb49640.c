// Function: sub_B49640
// Address: 0xb49640
//
__int64 __fastcall sub_B49640(__int64 a1, int a2, int a3)
{
  __int64 v3; // rbp
  __int64 v4; // rcx
  _QWORD v6[2]; // [rsp-10h] [rbp-10h] BYREF

  v4 = *(_QWORD *)(a1 - 32);
  if ( *(_BYTE *)v4 )
    return 0;
  v6[1] = v3;
  v6[0] = *(_QWORD *)(v4 + 120);
  return sub_A747F0(v6, a2 + 1, a3);
}
