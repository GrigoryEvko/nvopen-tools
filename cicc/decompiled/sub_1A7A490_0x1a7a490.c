// Function: sub_1A7A490
// Address: 0x1a7a490
//
bool __fastcall sub_1A7A490(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v3; // al
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdi
  bool result; // al
  unsigned __int64 v13; // rdx
  _QWORD *v14; // rdx

  v3 = *(_BYTE *)(a1 + 16);
  if ( v3 <= 0x10u )
    return 1;
  if ( v3 == 17 )
  {
    v6 = *(_QWORD *)(a2 + 40);
    v7 = *(_QWORD *)(v6 + 56);
    if ( (*(_BYTE *)(v7 + 18) & 1) != 0 )
      sub_15E08E0(*(_QWORD *)(v6 + 56), a2);
    v8 = *(_QWORD *)(v7 + 88);
    v9 = v8 == a1 ? 0LL : -858993459 * (unsigned int)((unsigned __int64)(a1 - 40 - v8) >> 3) + 1;
    v10 = *(_QWORD *)(a2 + 24 * (v9 - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    if ( v10 == a1 )
    {
      if ( v10 )
        return 1;
    }
  }
  v11 = sub_157F120(*(_QWORD *)(a3 + 40));
  result = 0;
  if ( v11 )
  {
    v13 = sub_157EBA0(v11);
    result = 0;
    if ( *(_BYTE *)(v13 + 16) == 27 )
    {
      if ( (*(_BYTE *)(v13 + 23) & 0x40) != 0 )
        v14 = *(_QWORD **)(v13 - 8);
      else
        v14 = (_QWORD *)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF));
      result = *v14 != 0 && *v14 == a1;
      if ( result )
        return *(_QWORD *)(a3 + 40) != v14[3];
    }
  }
  return result;
}
