// Function: sub_2B3C060
// Address: 0x2b3c060
//
bool __fastcall sub_2B3C060(__int64 a1, _DWORD *a2)
{
  __int64 v2; // r8
  int v3; // r9d
  _DWORD *v4; // rdi
  int v5; // r10d
  int v6; // r11d
  unsigned int i; // eax
  _DWORD *v8; // rcx
  unsigned int v9; // eax
  __int64 v10; // rdx

  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v2 = a1 + 16;
    v3 = 7;
    v4 = (_DWORD *)(a1 + 80);
  }
  else
  {
    v2 = *(_QWORD *)(a1 + 16);
    v10 = *(unsigned int *)(a1 + 24);
    v4 = (_DWORD *)(v2 + 8 * v10);
    if ( !(_DWORD)v10 )
      return 0;
    v3 = v10 - 1;
  }
  v5 = a2[1];
  v6 = 1;
  for ( i = v3
          & (((0xBF58476D1CE4E5B9LL * ((unsigned int)(37 * v5) | ((unsigned __int64)(unsigned int)(37 * *a2) << 32))) >> 31)
           ^ (756364221 * v5)); ; i = v3 & v9 )
  {
    v8 = (_DWORD *)(v2 + 8LL * i);
    if ( *a2 == *v8 && v5 == v8[1] )
      break;
    if ( *v8 == -1 && v8[1] == -1 )
      return 0;
    v9 = v6 + i;
    ++v6;
  }
  return v8 != v4;
}
