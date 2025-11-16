// Function: sub_15A6E90
// Address: 0x15a6e90
//
__int64 __fastcall sub_15A6E90(
        __int64 a1,
        __int64 a2,
        int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  int v10; // edx
  __int64 v13; // rdi
  __int64 v14; // r12

  v10 = 0;
  v13 = *(_QWORD *)(a1 + 8);
  if ( a8 )
    v10 = sub_161FF10(v13, a7, a8);
  v14 = sub_15BDB40(v13, 1, v10, 0, 0, 0, a4, a2, a3, 0, 0, a5, 0, 0, 0, 0, 0, 0, 1);
  sub_15A6B80(a1, v14);
  return v14;
}
