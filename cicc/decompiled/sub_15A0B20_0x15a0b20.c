// Function: sub_15A0B20
// Address: 0x15a0b20
//
__int64 __fastcall sub_15A0B20(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  int v5; // edx
  unsigned int v7; // r13d
  int v8; // r14d
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r12
  char v13; // al

  if ( *(_BYTE *)(a1 + 16) == 14 )
  {
    v4 = sub_16982C0(a1, a2, a3, a4);
    if ( *(_QWORD *)(a1 + 32) == v4 )
    {
      v4 = *(_QWORD *)(a1 + 40);
      v5 = *(_BYTE *)(v4 + 26) & 7;
      if ( (_BYTE)v5 == 1 )
        return 0;
    }
    else
    {
      v5 = *(_BYTE *)(a1 + 50) & 7;
      if ( (_BYTE)v5 == 1 )
        return 0;
    }
    LOBYTE(v4) = (_BYTE)v5 == 3;
    LOBYTE(v5) = (_BYTE)v5 == 0;
    return (v5 | (unsigned int)v4) ^ 1;
  }
  else
  {
    if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 16 )
      return 0;
    v7 = 0;
    v8 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
    if ( v8 )
    {
      while ( 1 )
      {
        v9 = sub_15A0A60(a1, v7);
        v12 = v9;
        if ( !v9 || *(_BYTE *)(v9 + 16) != 14 )
          break;
        if ( *(_QWORD *)(v9 + 32) == sub_16982C0(a1, v7, v10, v11) )
        {
          v13 = *(_BYTE *)(*(_QWORD *)(v12 + 40) + 26LL) & 7;
          if ( v13 == 1 )
            return 0;
        }
        else
        {
          v13 = *(_BYTE *)(v12 + 50) & 7;
          if ( v13 == 1 )
            return 0;
        }
        if ( v13 == 3 || !v13 )
          break;
        if ( ++v7 == v8 )
          return 1;
      }
      return 0;
    }
    return 1;
  }
}
