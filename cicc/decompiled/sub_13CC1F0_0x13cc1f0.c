// Function: sub_13CC1F0
// Address: 0x13cc1f0
//
__int64 __fastcall sub_13CC1F0(__int64 a1)
{
  unsigned int v1; // r12d
  unsigned __int8 v2; // al
  __int64 v3; // rbx
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // r13
  unsigned int v8; // r15d
  int v9; // r14d
  __int64 v10; // rax
  __int64 v11; // r13
  char v12; // al
  __int64 v13; // r13

  v2 = *(_BYTE *)(a1 + 16);
  if ( v2 == 14 )
  {
    if ( *(_QWORD *)(a1 + 32) == sub_16982C0() )
      v3 = *(_QWORD *)(a1 + 40) + 8LL;
    else
      v3 = a1 + 32;
    LOBYTE(v1) = (*(_BYTE *)(v3 + 18) & 7) == 3;
    return v1;
  }
  LOBYTE(v1) = v2 <= 0x10u && *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 16;
  if ( !(_BYTE)v1 )
    return 0;
  v5 = sub_15A1020();
  v6 = v5;
  if ( !v5 || *(_BYTE *)(v5 + 16) != 14 )
  {
    v8 = 0;
    v9 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
    if ( !v9 )
      return v1;
    while ( 1 )
    {
      v10 = sub_15A0A60(a1, v8);
      v11 = v10;
      if ( !v10 )
        break;
      v12 = *(_BYTE *)(v10 + 16);
      if ( v12 != 9 )
      {
        if ( v12 != 14 )
          break;
        v13 = *(_QWORD *)(v11 + 32) == sub_16982C0() ? *(_QWORD *)(v11 + 40) + 8LL : v11 + 32;
        if ( (*(_BYTE *)(v13 + 18) & 7) != 3 )
          break;
      }
      if ( v9 == ++v8 )
        return v1;
    }
    return 0;
  }
  if ( *(_QWORD *)(v5 + 32) == sub_16982C0() )
    v7 = *(_QWORD *)(v6 + 40) + 8LL;
  else
    v7 = v6 + 32;
  LOBYTE(v1) = (*(_BYTE *)(v7 + 18) & 7) == 3;
  return v1;
}
