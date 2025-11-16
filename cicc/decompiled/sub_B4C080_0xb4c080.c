// Function: sub_B4C080
// Address: 0xb4c080
//
__int64 __fastcall sub_B4C080(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rdx
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // rdx

  v2 = sub_BD5C60(a2, a2);
  v3 = sub_BCB120(v2);
  sub_B44260(a1, v3, 9, 2u, 0, 0);
  v4 = *(_QWORD *)(a2 - 64);
  if ( *(_QWORD *)(a1 - 64) )
  {
    v5 = *(_QWORD *)(a1 - 56);
    **(_QWORD **)(a1 - 48) = v5;
    if ( v5 )
      *(_QWORD *)(v5 + 16) = *(_QWORD *)(a1 - 48);
  }
  *(_QWORD *)(a1 - 64) = v4;
  if ( v4 )
  {
    v6 = *(_QWORD *)(v4 + 16);
    *(_QWORD *)(a1 - 56) = v6;
    if ( v6 )
      *(_QWORD *)(v6 + 16) = a1 - 56;
    *(_QWORD *)(a1 - 48) = v4 + 16;
    *(_QWORD *)(v4 + 16) = a1 - 64;
  }
  result = *(_QWORD *)(a2 - 32);
  if ( *(_QWORD *)(a1 - 32) )
  {
    v8 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v8;
    if ( v8 )
      *(_QWORD *)(v8 + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = result;
  if ( result )
  {
    v9 = *(_QWORD *)(result + 16);
    *(_QWORD *)(a1 - 24) = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = result + 16;
    *(_QWORD *)(result + 16) = a1 - 32;
  }
  return result;
}
