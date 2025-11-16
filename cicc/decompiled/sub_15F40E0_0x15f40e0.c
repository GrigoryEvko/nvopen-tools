// Function: sub_15F40E0
// Address: 0x15f40e0
//
char __fastcall sub_15F40E0(__int64 a1, __int64 a2)
{
  char v2; // r8
  __int64 v3; // rdx
  char result; // al
  _QWORD *v5; // r11
  __int64 v6; // rax
  _QWORD *v7; // r10
  _QWORD *v8; // r9
  _QWORD *v9; // rcx
  _QWORD *v10; // rax

  v2 = *(_BYTE *)(a1 + 16);
  if ( v2 != *(_BYTE *)(a2 + 16) )
    return 0;
  v3 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( (_DWORD)v3 != (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) )
    return 0;
  result = 0;
  if ( *(_QWORD *)a1 == *(_QWORD *)a2 )
  {
    if ( !(_DWORD)v3 )
      return sub_15F3E20(a1, a2, 0);
    v5 = (*(_BYTE *)(a2 + 23) & 0x40) != 0 ? *(_QWORD **)(a2 - 8) : (_QWORD *)(a2 - 24 * v3);
    v6 = 3 * v3;
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    {
      v7 = *(_QWORD **)(a1 - 8);
      v8 = &v7[v6];
    }
    else
    {
      v8 = (_QWORD *)a1;
      v7 = (_QWORD *)(a1 - v6 * 8);
    }
    v9 = v5;
    v10 = v7;
    do
    {
      if ( *v10 != *v9 )
        return 0;
      v10 += 3;
      v9 += 3;
    }
    while ( v10 != v8 );
    if ( v2 == 77 )
      return memcmp(&v7[3 * *(unsigned int *)(a1 + 56) + 1], &v5[3 * *(unsigned int *)(a2 + 56) + 1], 8 * v3) == 0;
    else
      return sub_15F3E20(a1, a2, 0);
  }
  return result;
}
