// Function: sub_EC5140
// Address: 0xec5140
//
bool __fastcall sub_EC5140(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 v3; // rax
  int v5; // eax

  v1 = *(_QWORD *)(a1 + 16);
  v2 = *(_QWORD *)(a1 + 8);
  if ( *(_DWORD *)a1 != 2 )
  {
    if ( !v1 )
      return 0;
    v3 = v1 - 1;
    if ( !v3 )
      v3 = 1;
    ++v2;
    v1 = v3 - 1;
  }
  if ( v1 != 11 )
    return 0;
  if ( *(_QWORD *)v2 != 0x737265765F6B6473LL || *(_WORD *)(v2 + 8) != 28521 || (v5 = 0, *(_BYTE *)(v2 + 10) != 110) )
    v5 = 1;
  return v5 == 0;
}
