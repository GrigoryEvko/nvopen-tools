// Function: sub_ADE1A0
// Address: 0xade1a0
//
__int64 __fastcall sub_ADE1A0(
        __int64 a1,
        _BYTE *a2,
        __int64 a3,
        __int64 a4,
        int a5,
        int a6,
        __int64 a7,
        int a8,
        __int64 a9,
        __int64 a10)
{
  int v13; // ebx
  __int64 v14; // rdi
  int v15; // r10d
  __int64 v16; // r13

  v13 = (int)a2;
  if ( a2 && *a2 == 17 )
    v13 = 0;
  v14 = *(_QWORD *)(a1 + 8);
  v15 = 0;
  if ( a4 )
    v15 = sub_B9B140(v14, a3, a4);
  v16 = sub_B065E0(v14, 1, v15, a5, a6, v13, a9, a7, a8, 0, 0, a10, 0);
  sub_ADDDC0(a1, v16);
  return v16;
}
