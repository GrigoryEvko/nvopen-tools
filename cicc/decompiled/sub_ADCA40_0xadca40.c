// Function: sub_ADCA40
// Address: 0xadca40
//
__int64 __fastcall sub_ADCA40(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        int a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  int v10; // edx
  __int64 v11; // r15

  v10 = 0;
  v11 = *(_QWORD *)(a1 + 8);
  if ( a8 )
    v10 = sub_B9B140(v11, a7, a8);
  return sub_B05AE0(v11, 15, v10, 0, 0, 0, a2, a3, a4, 0);
}
