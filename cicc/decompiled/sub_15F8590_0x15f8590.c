// Function: sub_15F8590
// Address: 0x15f8590
//
unsigned __int64 __fastcall sub_15F8590(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  unsigned __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rax

  v4 = sub_157E9C0(a2);
  v5 = sub_1643270(v4);
  result = sub_15F1F50(a1, v5, 2, a1 - 24, 1, a3);
  if ( *(_QWORD *)(a1 - 24) )
  {
    v7 = *(_QWORD *)(a1 - 16);
    result = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)result = v7;
    if ( v7 )
    {
      result |= *(_QWORD *)(v7 + 16) & 3LL;
      *(_QWORD *)(v7 + 16) = result;
    }
  }
  *(_QWORD *)(a1 - 24) = a2;
  if ( a2 )
  {
    v8 = *(_QWORD *)(a2 + 8);
    *(_QWORD *)(a1 - 16) = v8;
    if ( v8 )
      *(_QWORD *)(v8 + 16) = (a1 - 16) | *(_QWORD *)(v8 + 16) & 3LL;
    v9 = *(_QWORD *)(a1 - 8);
    *(_QWORD *)(a2 + 8) = a1 - 24;
    result = (a2 + 8) | v9 & 3;
    *(_QWORD *)(a1 - 8) = result;
  }
  return result;
}
