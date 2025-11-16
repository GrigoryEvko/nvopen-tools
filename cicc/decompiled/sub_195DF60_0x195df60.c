// Function: sub_195DF60
// Address: 0x195df60
//
__int64 __fastcall sub_195DF60(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdi
  int v4; // ecx
  __int64 v5; // rdx
  unsigned int v6; // r8d
  unsigned int v7; // eax
  __int64 v8; // rsi
  int v10; // eax
  int i; // r8d

  v2 = *(_QWORD *)(a1 + 24);
  if ( (*(_BYTE *)(v2 + 8) & 1) != 0 )
  {
    v3 = v2 + 16;
    v4 = 7;
  }
  else
  {
    v3 = *(_QWORD *)(v2 + 16);
    v10 = *(_DWORD *)(v2 + 24);
    v6 = 0;
    if ( !v10 )
      return v6;
    v4 = v10 - 1;
  }
  v5 = *(_QWORD *)(a2 - 24);
  v6 = 1;
  v7 = v4 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v8 = *(_QWORD *)(v3 + 8LL * v7);
  if ( v5 != v8 )
  {
    for ( i = 1; ; ++i )
    {
      if ( v8 == -8 )
        return 0;
      v7 = v4 & (i + v7);
      v8 = *(_QWORD *)(v3 + 8LL * v7);
      if ( v8 == v5 )
        break;
    }
    return 1;
  }
  return v6;
}
