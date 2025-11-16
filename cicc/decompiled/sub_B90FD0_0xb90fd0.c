// Function: sub_B90FD0
// Address: 0xb90fd0
//
void __fastcall sub_B90FD0(__int64 a1, __int64 a2)
{
  __int64 v2; // r9
  int v3; // edx
  unsigned int v4; // eax
  __int64 *v5; // rcx
  __int64 v6; // r8
  unsigned int v7; // eax
  int v8; // edx
  int v9; // ecx
  int v10; // r10d

  if ( (*(_BYTE *)(a1 + 24) & 1) != 0 )
  {
    v2 = a1 + 32;
    v3 = 3;
  }
  else
  {
    v8 = *(_DWORD *)(a1 + 40);
    v2 = *(_QWORD *)(a1 + 32);
    if ( !v8 )
      return;
    v3 = v8 - 1;
  }
  v4 = v3 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v5 = (__int64 *)(v2 + 24LL * v4);
  v6 = *v5;
  if ( *v5 == a2 )
  {
LABEL_4:
    *v5 = -8192;
    v7 = *(_DWORD *)(a1 + 24);
    ++*(_DWORD *)(a1 + 28);
    *(_DWORD *)(a1 + 24) = (2 * (v7 >> 1) - 2) | v7 & 1;
  }
  else
  {
    v9 = 1;
    while ( v6 != -4096 )
    {
      v10 = v9 + 1;
      v4 = v3 & (v9 + v4);
      v5 = (__int64 *)(v2 + 24LL * v4);
      v6 = *v5;
      if ( *v5 == a2 )
        goto LABEL_4;
      v9 = v10;
    }
  }
}
