// Function: sub_F8E490
// Address: 0xf8e490
//
__int64 __fastcall sub_F8E490(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  int v3; // ecx
  unsigned int v4; // edx
  __int64 *v5; // rax
  __int64 v6; // rsi
  int v8; // ecx
  int v9; // eax
  int v10; // r9d

  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v2 = a2 + 16;
    v3 = 3;
  }
  else
  {
    v8 = *(_DWORD *)(a2 + 24);
    v2 = *(_QWORD *)(a2 + 16);
    if ( !v8 )
      return 0;
    v3 = v8 - 1;
  }
  v4 = v3 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v5 = (__int64 *)(v2 + 16LL * v4);
  v6 = *v5;
  if ( a1 == *v5 )
    return v5[1];
  v9 = 1;
  while ( v6 != -4096 )
  {
    v10 = v9 + 1;
    v4 = v3 & (v9 + v4);
    v5 = (__int64 *)(v2 + 16LL * v4);
    v6 = *v5;
    if ( a1 == *v5 )
      return v5[1];
    v9 = v10;
  }
  return 0;
}
