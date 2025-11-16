// Function: sub_102AC30
// Address: 0x102ac30
//
void __fastcall sub_102AC30(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  int v4; // ecx
  unsigned int v5; // eax
  __int64 *v6; // r12
  __int64 v7; // rdx
  unsigned int v8; // eax
  int v9; // ecx
  int v10; // r8d

  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v3 = a1 + 16;
    v4 = 3;
  }
  else
  {
    v9 = *(_DWORD *)(a1 + 24);
    v3 = *(_QWORD *)(a1 + 16);
    if ( !v9 )
      return;
    v4 = v9 - 1;
  }
  v5 = v4 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v6 = (__int64 *)(v3 + 88LL * v5);
  v7 = *v6;
  if ( *v6 == a2 )
  {
LABEL_4:
    if ( (v6[2] & 1) == 0 )
      sub_C7D6A0(v6[3], 16LL * *((unsigned int *)v6 + 8), 8);
    *v6 = -8192;
    v8 = *(_DWORD *)(a1 + 8);
    ++*(_DWORD *)(a1 + 12);
    *(_DWORD *)(a1 + 8) = (2 * (v8 >> 1) - 2) | v8 & 1;
  }
  else
  {
    v10 = 1;
    while ( v7 != -4096 )
    {
      v5 = v4 & (v10 + v5);
      v6 = (__int64 *)(v3 + 88LL * v5);
      v7 = *v6;
      if ( *v6 == a2 )
        goto LABEL_4;
      ++v10;
    }
  }
}
