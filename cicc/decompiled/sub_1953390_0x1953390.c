// Function: sub_1953390
// Address: 0x1953390
//
bool __fastcall sub_1953390(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rdx
  int v9; // ebx

  v2 = sub_157EBA0(a2);
  v3 = v2;
  if ( !*(_QWORD *)(v2 + 48) && *(__int16 *)(v2 + 18) >= 0 )
    return 0;
  v4 = sub_1625790(v2, 2);
  v5 = v4;
  if ( !v4 )
    return 0;
  v6 = sub_161E970(*(_QWORD *)(v4 - 8LL * *(unsigned int *)(v4 + 8)));
  if ( v7 != 14
    || *(_QWORD *)v6 != 0x775F68636E617262LL
    || *(_DWORD *)(v6 + 8) != 1751607653
    || *(_WORD *)(v6 + 12) != 29556 )
  {
    return 0;
  }
  v9 = *(_DWORD *)(v5 + 8);
  return (unsigned int)sub_15F4D60(v3) + 1 == v9;
}
