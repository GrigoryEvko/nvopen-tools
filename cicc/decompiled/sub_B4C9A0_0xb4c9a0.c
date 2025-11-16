// Function: sub_B4C9A0
// Address: 0xb4c9a0
//
_QWORD *__fastcall sub_B4C9A0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v11; // rax
  __int64 v12; // rax
  _QWORD *result; // rax
  __int64 v14; // rax
  __int64 v15; // rax

  v11 = sub_AA48A0(a2);
  v12 = sub_BCB120(v11);
  result = sub_B44260(a1, v12, 2, a5, a7, a8);
  if ( *(_QWORD *)(a1 - 96) )
  {
    result = *(_QWORD **)(a1 - 88);
    **(_QWORD **)(a1 - 80) = result;
    if ( result )
      result[2] = *(_QWORD *)(a1 - 80);
  }
  *(_QWORD *)(a1 - 96) = a4;
  if ( a4 )
  {
    v14 = *(_QWORD *)(a4 + 16);
    *(_QWORD *)(a1 - 88) = v14;
    if ( v14 )
      *(_QWORD *)(v14 + 16) = a1 - 88;
    result = (_QWORD *)(a1 - 96);
    *(_QWORD *)(a1 - 80) = a4 + 16;
    *(_QWORD *)(a4 + 16) = a1 - 96;
  }
  if ( *(_QWORD *)(a1 - 64) )
  {
    result = *(_QWORD **)(a1 - 56);
    **(_QWORD **)(a1 - 48) = result;
    if ( result )
      result[2] = *(_QWORD *)(a1 - 48);
  }
  *(_QWORD *)(a1 - 64) = a3;
  if ( a3 )
  {
    v15 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(a1 - 56) = v15;
    if ( v15 )
      *(_QWORD *)(v15 + 16) = a1 - 56;
    result = (_QWORD *)(a1 - 64);
    *(_QWORD *)(a1 - 48) = a3 + 16;
    *(_QWORD *)(a3 + 16) = a1 - 64;
  }
  if ( *(_QWORD *)(a1 - 32) )
  {
    result = *(_QWORD **)(a1 - 24);
    **(_QWORD **)(a1 - 16) = result;
    if ( result )
      result[2] = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = a2;
  if ( a2 )
  {
    result = *(_QWORD **)(a2 + 16);
    *(_QWORD *)(a1 - 24) = result;
    if ( result )
      result[2] = a1 - 24;
    *(_QWORD *)(a1 - 16) = a2 + 16;
    *(_QWORD *)(a2 + 16) = a1 - 32;
  }
  return result;
}
