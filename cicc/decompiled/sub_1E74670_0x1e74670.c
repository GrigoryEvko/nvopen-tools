// Function: sub_1E74670
// Address: 0x1e74670
//
void __fastcall sub_1E74670(__int64 a1, __int64 a2, char a3)
{
  unsigned int v4; // eax
  bool v5; // cf
  __int64 v6; // rdi
  char v7; // dl
  unsigned int v8; // eax
  __int64 v9; // rdi

  if ( a3 )
  {
    v4 = *(_DWORD *)(a2 + 248);
    v5 = *(_DWORD *)(a1 + 308) < v4;
    v6 = a1 + 144;
    if ( !v5 )
      v4 = *(_DWORD *)(v6 + 164);
    *(_DWORD *)(a2 + 248) = v4;
    sub_1E73130(v6, a2);
    v7 = 1;
    if ( (*(_BYTE *)(a2 + 228) & 0x20) == 0 )
      return;
LABEL_10:
    sub_1E74580(a1, (_QWORD *)a2, v7);
    return;
  }
  v8 = *(_DWORD *)(a1 + 676);
  v9 = a1 + 512;
  if ( *(_DWORD *)(a2 + 252) >= v8 )
    v8 = *(_DWORD *)(a2 + 252);
  *(_DWORD *)(a2 + 252) = v8;
  sub_1E73130(v9, a2);
  if ( (*(_BYTE *)(a2 + 228) & 0x40) != 0 )
  {
    v7 = 0;
    goto LABEL_10;
  }
}
