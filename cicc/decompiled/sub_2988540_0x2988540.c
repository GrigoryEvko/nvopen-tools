// Function: sub_2988540
// Address: 0x2988540
//
__int64 __fastcall sub_2988540(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // rsi
  int v4; // edx
  unsigned int v5; // eax
  __int64 v6; // rdi
  int v8; // edx
  int v9; // r8d

  v1 = a1[1];
  v2 = *a1;
  if ( (*(_BYTE *)(v1 + 8) & 1) != 0 )
  {
    v3 = v1 + 16;
    v4 = 3;
  }
  else
  {
    v8 = *(_DWORD *)(v1 + 24);
    v3 = *(_QWORD *)(v1 + 16);
    if ( !v8 )
      return 0;
    v4 = v8 - 1;
  }
  v5 = v4 & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
  v6 = *(_QWORD *)(v3 + 8LL * v5);
  if ( v2 == v6 )
    return 1;
  v9 = 1;
  while ( v6 != -4096 )
  {
    v5 = v4 & (v9 + v5);
    v6 = *(_QWORD *)(v3 + 8LL * v5);
    if ( v2 == v6 )
      return 1;
    ++v9;
  }
  return 0;
}
