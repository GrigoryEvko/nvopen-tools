// Function: sub_B523C0
// Address: 0xb523c0
//
__int64 __fastcall sub_B523C0(
        __int64 a1,
        __int64 a2,
        int a3,
        __int16 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10)
{
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 result; // rax

  sub_B44260(a1, a2, a3, 2u, a8, a9);
  if ( *(_QWORD *)(a1 - 64) )
  {
    v12 = *(_QWORD *)(a1 - 56);
    **(_QWORD **)(a1 - 48) = v12;
    if ( v12 )
      *(_QWORD *)(v12 + 16) = *(_QWORD *)(a1 - 48);
  }
  *(_QWORD *)(a1 - 64) = a5;
  if ( a5 )
  {
    v13 = *(_QWORD *)(a5 + 16);
    *(_QWORD *)(a1 - 56) = v13;
    if ( v13 )
      *(_QWORD *)(v13 + 16) = a1 - 56;
    *(_QWORD *)(a1 - 48) = a5 + 16;
    *(_QWORD *)(a5 + 16) = a1 - 64;
  }
  if ( *(_QWORD *)(a1 - 32) )
  {
    v14 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v14;
    if ( v14 )
      *(_QWORD *)(v14 + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = a6;
  if ( a6 )
  {
    v15 = *(_QWORD *)(a6 + 16);
    *(_QWORD *)(a1 - 24) = v15;
    if ( v15 )
      *(_QWORD *)(v15 + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = a6 + 16;
    *(_QWORD *)(a6 + 16) = a1 - 32;
  }
  *(_WORD *)(a1 + 2) = a4 | *(_WORD *)(a1 + 2) & 0xFFC0;
  result = sub_BD6B50(a1, a7);
  if ( a10 )
    return sub_B45260((unsigned __int8 *)a1, a10, 1);
  return result;
}
