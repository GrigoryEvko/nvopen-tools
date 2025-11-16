// Function: sub_B4BC30
// Address: 0xb4bc30
//
__int64 __fastcall sub_B4BC30(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // rdx

  v2 = sub_BD5C60(a2, a2);
  v3 = sub_BCB120(v2);
  sub_B44260(a1, v3, 6, 1u, 0, 0);
  result = *(_QWORD *)(a2 - 32);
  if ( *(_QWORD *)(a1 - 32) )
  {
    v5 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v5;
    if ( v5 )
      *(_QWORD *)(v5 + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = result;
  if ( result )
  {
    v6 = *(_QWORD *)(result + 16);
    *(_QWORD *)(a1 - 24) = v6;
    if ( v6 )
      *(_QWORD *)(v6 + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = result + 16;
    *(_QWORD *)(result + 16) = a1 - 32;
  }
  return result;
}
