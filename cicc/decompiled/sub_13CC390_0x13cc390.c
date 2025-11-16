// Function: sub_13CC390
// Address: 0x13cc390
//
__int64 __fastcall sub_13CC390(__int64 a1)
{
  unsigned int v1; // r12d
  unsigned __int8 v2; // al
  __int64 v3; // rbx
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // rax
  unsigned int v9; // r15d
  int v10; // r14d
  __int64 v11; // rax
  __int64 v12; // r13
  char v13; // al
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // rax

  v2 = *(_BYTE *)(a1 + 16);
  if ( v2 == 14 )
  {
    if ( *(_QWORD *)(a1 + 32) == sub_16982C0() )
    {
      v5 = *(_QWORD *)(a1 + 40);
      if ( (*(_BYTE *)(v5 + 26) & 7) != 3 )
        return 0;
      v3 = v5 + 8;
    }
    else
    {
      v3 = a1 + 32;
      if ( (*(_BYTE *)(a1 + 50) & 7) != 3 )
        return 0;
    }
    return (*(_BYTE *)(v3 + 18) & 8) != 0;
  }
  LOBYTE(v1) = v2 <= 0x10u && *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 16;
  if ( (_BYTE)v1 )
  {
    v6 = sub_15A1020();
    v7 = v6;
    if ( !v6 || *(_BYTE *)(v6 + 16) != 14 )
    {
      v9 = 0;
      v10 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
      if ( !v10 )
        return v1;
      while ( 1 )
      {
        v11 = sub_15A0A60(a1, v9);
        v12 = v11;
        if ( !v11 )
          break;
        v13 = *(_BYTE *)(v11 + 16);
        if ( v13 != 9 )
        {
          if ( v13 != 14 )
            return 0;
          if ( *(_QWORD *)(v12 + 32) == sub_16982C0() )
          {
            v15 = *(_QWORD *)(v12 + 40);
            if ( (*(_BYTE *)(v15 + 26) & 7) != 3 )
              return 0;
            v14 = v15 + 8;
          }
          else
          {
            if ( (*(_BYTE *)(v12 + 50) & 7) != 3 )
              return 0;
            v14 = v12 + 32;
          }
          if ( (*(_BYTE *)(v14 + 18) & 8) == 0 )
            return 0;
        }
        if ( v10 == ++v9 )
          return v1;
      }
      return 0;
    }
    if ( *(_QWORD *)(v6 + 32) == sub_16982C0() )
    {
      v16 = *(_QWORD *)(v7 + 40);
      if ( (*(_BYTE *)(v16 + 26) & 7) != 3 )
        return 0;
      v8 = v16 + 8;
    }
    else
    {
      if ( (*(_BYTE *)(v7 + 50) & 7) != 3 )
        return 0;
      v8 = v7 + 32;
    }
    return (*(_BYTE *)(v8 + 18) & 8) != 0;
  }
  return 0;
}
