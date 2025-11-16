// Function: sub_AA60B0
// Address: 0xaa60b0
//
__int64 __fastcall sub_AA60B0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rsi
  int v4; // ecx
  unsigned int v5; // edx
  __int64 *v6; // rax
  __int64 v7; // rdi
  int v9; // ecx
  int v10; // eax
  int v11; // r8d

  v2 = *(_QWORD *)sub_AA48A0(a1);
  if ( (*(_BYTE *)(v2 + 3520) & 1) != 0 )
  {
    v3 = v2 + 3528;
    v4 = 3;
  }
  else
  {
    v9 = *(_DWORD *)(v2 + 3536);
    v3 = *(_QWORD *)(v2 + 3528);
    if ( !v9 )
      return 0;
    v4 = v9 - 1;
  }
  v5 = v4 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v6 = (__int64 *)(v3 + 16LL * v5);
  v7 = *v6;
  if ( a1 == *v6 )
    return v6[1];
  v10 = 1;
  while ( v7 != -4096 )
  {
    v11 = v10 + 1;
    v5 = v4 & (v10 + v5);
    v6 = (__int64 *)(v3 + 16LL * v5);
    v7 = *v6;
    if ( a1 == *v6 )
      return v6[1];
    v10 = v11;
  }
  return 0;
}
