// Function: sub_1F3CFB0
// Address: 0x1f3cfb0
//
__int64 __fastcall sub_1F3CFB0(__int64 a1, unsigned __int8 a2, __int64 a3, char a4)
{
  unsigned __int8 v4; // al
  unsigned int v5; // r8d
  __int64 v7; // rdx
  __int64 v8; // r9
  unsigned int v9; // r8d

  v4 = a2;
  if ( !a2 || !a4 || *(_BYTE *)(a1 + 259LL * a2 + 2607) != 1 )
    return 1;
  v7 = *(_QWORD *)(a1 + 74064);
  if ( !v7 )
    goto LABEL_18;
  v8 = a1 + 74056;
  do
  {
    v9 = *(_DWORD *)(v7 + 32);
    if ( v9 <= 0xB8 || v9 == 185 && a2 > *(_BYTE *)(v7 + 36) )
    {
      v7 = *(_QWORD *)(v7 + 24);
    }
    else
    {
      v8 = v7;
      v7 = *(_QWORD *)(v7 + 16);
    }
  }
  while ( v7 );
  if ( a1 + 74056 == v8 || *(_DWORD *)(v8 + 32) > 0xB9u || *(_DWORD *)(v8 + 32) == 185 && a2 < *(_BYTE *)(v8 + 36) )
  {
LABEL_18:
    do
    {
      do
        ++v4;
      while ( !v4 );
    }
    while ( !*(_QWORD *)(a1 + 8LL * v4 + 120) || *(_BYTE *)(a1 + 259LL * v4 + 2607) == 1 );
  }
  else
  {
    v4 = *(_BYTE *)(v8 + 40);
  }
  v5 = 0;
  if ( a4 != v4 )
    return 1;
  return v5;
}
