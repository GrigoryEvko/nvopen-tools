// Function: sub_15A5B80
// Address: 0x15a5b80
//
__int64 __fastcall sub_15A5B80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6, _BYTE *a7)
{
  int v9; // ebx
  __int64 v10; // r15
  int v11; // r10d
  _BYTE v13[56]; // [rsp+8h] [rbp-38h] BYREF

  v9 = (int)a7;
  if ( a7 && *a7 == 16 )
    v9 = 0;
  v13[4] = 0;
  v10 = *(_QWORD *)(a1 + 8);
  v11 = 0;
  if ( a4 )
    v11 = sub_161FF10(*(_QWORD *)(a1 + 8), a3, a4);
  return sub_15BD310(v10, 22, v11, a5, a6, v9, a2, 0, 0, 0, (__int64)v13, 0, 0, 0, 1);
}
