// Function: sub_2598010
// Address: 0x2598010
//
__int64 __fastcall sub_2598010(__int64 a1, __int64 a2)
{
  int v2; // ebx
  int v3; // ebx
  char v5; // [rsp+Fh] [rbp-41h] BYREF
  __int64 v6[8]; // [rsp+10h] [rbp-40h] BYREF

  v6[0] = a2;
  v6[1] = sub_250D070((_QWORD *)(a1 + 72));
  v5 = 0;
  v6[2] = a1;
  v6[3] = (__int64)&v5;
  v2 = sub_2597910(v6, a1 + 104, 1u);
  v3 = sub_2597910(v6, a1 + 216, 2u) | v2;
  if ( !v5 )
    *(_BYTE *)(a1 + 96) = *(_BYTE *)(a1 + 97);
  return (unsigned __int8)v3 ^ 1u;
}
