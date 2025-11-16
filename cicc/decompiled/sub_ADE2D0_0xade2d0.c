// Function: sub_ADE2D0
// Address: 0xade2d0
//
__int64 __fastcall sub_ADE2D0(
        __int64 a1,
        int a2,
        __int64 a3,
        __int64 a4,
        _BYTE *a5,
        int a6,
        int a7,
        int a8,
        __int64 a9,
        int a10,
        int a11,
        __int64 a12,
        __int64 a13)
{
  __int64 v13; // r10
  int v17; // ebx
  __int64 v18; // rdi
  int v19; // edx
  __int64 v20; // r12

  v13 = a3;
  v17 = (int)a5;
  if ( a5 && *a5 == 17 )
    v17 = 0;
  v18 = *(_QWORD *)(a1 + 8);
  if ( a13 )
  {
    sub_B9B140(v18, a12, a13);
    v13 = a3;
  }
  v19 = 0;
  if ( a4 )
    v19 = sub_B9B140(v18, v13, a4);
  v20 = sub_B065E0(v18, a2, v19, a6, a7, v17, 0, a9, a10, 0, a11, 0, a8);
  sub_ADDDC0(a1, v20);
  return v20;
}
