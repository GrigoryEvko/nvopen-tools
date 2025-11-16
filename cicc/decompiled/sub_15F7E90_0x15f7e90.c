// Function: sub_15F7E90
// Address: 0x15f7e90
//
__int64 __fastcall sub_15F7E90(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  _QWORD *i; // r9
  __int64 v5; // rax
  __int64 v6; // rcx
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  __int64 result; // rax

  v2 = 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  v3 = a1 - v2;
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v3 = *(_QWORD *)(a1 - 8);
  for ( i = (_QWORD *)(v3 + v2 - 24); i != a2; a2 += 3 )
  {
    v5 = a2[3];
    if ( *a2 )
    {
      v6 = a2[1];
      v7 = a2[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v7 = v6;
      if ( v6 )
        *(_QWORD *)(v6 + 16) = *(_QWORD *)(v6 + 16) & 3LL | v7;
    }
    *a2 = v5;
    if ( v5 )
    {
      v8 = *(_QWORD *)(v5 + 8);
      a2[1] = v8;
      if ( v8 )
        *(_QWORD *)(v8 + 16) = (unsigned __int64)(a2 + 1) | *(_QWORD *)(v8 + 16) & 3LL;
      a2[2] = (v5 + 8) | a2[2] & 3LL;
      *(_QWORD *)(v5 + 8) = a2;
    }
  }
  if ( *i )
  {
    v9 = i[1];
    v10 = i[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v10 = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = *(_QWORD *)(v9 + 16) & 3LL | v10;
  }
  *i = 0;
  result = (*(_DWORD *)(a1 + 20) + 0xFFFFFFF) & 0xFFFFFFF | *(_DWORD *)(a1 + 20) & 0xF0000000;
  *(_DWORD *)(a1 + 20) = result;
  return result;
}
