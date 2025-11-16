// Function: sub_15F7830
// Address: 0x15f7830
//
__int64 __fastcall sub_15F7830(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rcx
  unsigned __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 result; // rax
  __int64 v9; // rcx
  unsigned __int64 v10; // rdx
  __int64 v11; // rdx

  v2 = sub_16498A0(a2);
  v3 = sub_1643270(v2);
  sub_15F1EA0(a1, v3, 9, a1 - 48, 2, 0);
  v4 = *(_QWORD *)(a2 - 48);
  if ( *(_QWORD *)(a1 - 48) )
  {
    v5 = *(_QWORD *)(a1 - 40);
    v6 = *(_QWORD *)(a1 - 32) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v6 = v5;
    if ( v5 )
      *(_QWORD *)(v5 + 16) = *(_QWORD *)(v5 + 16) & 3LL | v6;
  }
  *(_QWORD *)(a1 - 48) = v4;
  if ( v4 )
  {
    v7 = *(_QWORD *)(v4 + 8);
    *(_QWORD *)(a1 - 40) = v7;
    if ( v7 )
      *(_QWORD *)(v7 + 16) = (a1 - 40) | *(_QWORD *)(v7 + 16) & 3LL;
    *(_QWORD *)(a1 - 32) = (v4 + 8) | *(_QWORD *)(a1 - 32) & 3LL;
    *(_QWORD *)(v4 + 8) = a1 - 48;
  }
  result = *(_QWORD *)(a2 - 24);
  if ( *(_QWORD *)(a1 - 24) )
  {
    v9 = *(_QWORD *)(a1 - 16);
    v10 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v10 = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = *(_QWORD *)(v9 + 16) & 3LL | v10;
  }
  *(_QWORD *)(a1 - 24) = result;
  if ( result )
  {
    v11 = *(_QWORD *)(result + 8);
    *(_QWORD *)(a1 - 16) = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = (a1 - 16) | *(_QWORD *)(v11 + 16) & 3LL;
    *(_QWORD *)(a1 - 24 + 16) = (result + 8) | *(_QWORD *)(a1 - 8) & 3LL;
    *(_QWORD *)(result + 8) = a1 - 24;
  }
  return result;
}
