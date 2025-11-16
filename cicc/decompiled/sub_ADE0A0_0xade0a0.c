// Function: sub_ADE0A0
// Address: 0xade0a0
//
__int64 __fastcall sub_ADE0A0(
        __int64 a1,
        _BYTE *a2,
        __int64 a3,
        __int64 a4,
        int a5,
        int a6,
        __int64 a7,
        int a8,
        int a9,
        __int64 a10,
        __int64 a11,
        int a12,
        __int64 a13,
        __int64 a14,
        __int64 a15)
{
  __int64 v15; // r10
  int v20; // ebx
  __int64 v21; // rdi
  int v22; // edx
  __int64 v23; // r12

  v15 = a3;
  v20 = (int)a2;
  if ( a2 && *a2 == 17 )
    v20 = 0;
  v21 = *(_QWORD *)(a1 + 8);
  if ( a15 )
  {
    sub_B9B140(v21, a14, a15);
    v15 = a3;
  }
  v22 = 0;
  if ( a4 )
    v22 = sub_B9B140(v21, v15, a4);
  v23 = sub_B065E0(v21, 19, v22, a5, a6, v20, a10, a7, a8, 0, a9, a11, a12);
  sub_ADDDC0(a1, v23);
  return v23;
}
