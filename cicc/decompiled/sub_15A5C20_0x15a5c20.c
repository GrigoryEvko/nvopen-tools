// Function: sub_15A5C20
// Address: 0x15a5c20
//
__int64 __fastcall sub_15A5C20(
        __int64 a1,
        _BYTE *a2,
        __int64 a3,
        __int64 a4,
        int a5,
        int a6,
        __int64 a7,
        int a8,
        __int64 a9,
        int a10,
        __int64 a11)
{
  int v13; // ebx
  __int64 v14; // r14
  int v15; // r10d
  _BYTE v17[40]; // [rsp+8h] [rbp-28h] BYREF

  v13 = (int)a2;
  if ( a2 && *a2 == 16 )
    v13 = 0;
  v17[4] = 0;
  v14 = *(_QWORD *)(a1 + 8);
  v15 = 0;
  if ( a4 )
    v15 = sub_161FF10(*(_QWORD *)(a1 + 8), a3, a4);
  return sub_15BD310(v14, 13, v15, a5, a6, v13, a11, a7, a8, a9, (__int64)v17, a10, 0, 0, 1);
}
