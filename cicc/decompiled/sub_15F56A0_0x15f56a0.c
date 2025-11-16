// Function: sub_15F56A0
// Address: 0x15f56a0
//
__int64 __fastcall sub_15F56A0(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // rsi
  __int64 v3; // rcx
  __int64 v4; // rax
  __int64 v5; // rax

  if ( (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) == 0 )
    return 1;
  v1 = 0;
  v2 = 0;
  v3 = 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  while ( 1 )
  {
    v4 = a1 - v3;
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v4 = *(_QWORD *)(a1 - 8);
    v5 = *(_QWORD *)(v4 + v1);
    if ( !v5 )
      BUG();
    if ( a1 == v5 || *(_BYTE *)(v5 + 16) == 9 )
      goto LABEL_11;
    if ( v5 != v2 && v2 )
      return 0;
    v2 = v5;
LABEL_11:
    v1 += 24;
    if ( v1 == v3 )
      return 1;
  }
}
