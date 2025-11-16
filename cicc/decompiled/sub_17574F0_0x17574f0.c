// Function: sub_17574F0
// Address: 0x17574f0
//
__int64 __fastcall sub_17574F0(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r12d
  unsigned __int8 v5; // al
  _BYTE *v6; // rbx
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // rax
  unsigned int v12; // r15d
  int v13; // r14d
  __int64 v14; // rax
  __int64 v15; // r13
  char v16; // al
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // rax

  v5 = a1[16];
  if ( v5 == 14 )
  {
    if ( *((void **)a1 + 4) == sub_16982C0() )
    {
      v8 = *((_QWORD *)a1 + 5);
      if ( (*(_BYTE *)(v8 + 26) & 7) != 3 )
        return 0;
      v6 = (_BYTE *)(v8 + 8);
    }
    else
    {
      v6 = a1 + 32;
      if ( (a1[50] & 7) != 3 )
        return 0;
    }
    return ((v6[18] >> 3) ^ 1) & 1;
  }
  LOBYTE(v4) = v5 <= 0x10u && *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 16;
  if ( (_BYTE)v4 )
  {
    v9 = sub_15A1020(a1, a2, *(_QWORD *)a1, a4);
    v10 = v9;
    if ( !v9 || *(_BYTE *)(v9 + 16) != 14 )
    {
      v12 = 0;
      v13 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
      if ( !v13 )
        return v4;
      while ( 1 )
      {
        v14 = sub_15A0A60((__int64)a1, v12);
        v15 = v14;
        if ( !v14 )
          break;
        v16 = *(_BYTE *)(v14 + 16);
        if ( v16 != 9 )
        {
          if ( v16 != 14 )
            return 0;
          if ( *(void **)(v15 + 32) == sub_16982C0() )
          {
            v18 = *(_QWORD *)(v15 + 40);
            if ( (*(_BYTE *)(v18 + 26) & 7) != 3 )
              return 0;
            v17 = v18 + 8;
          }
          else
          {
            if ( (*(_BYTE *)(v15 + 50) & 7) != 3 )
              return 0;
            v17 = v15 + 32;
          }
          if ( (*(_BYTE *)(v17 + 18) & 8) != 0 )
            return 0;
        }
        if ( v13 == ++v12 )
          return v4;
      }
      return 0;
    }
    if ( *(void **)(v9 + 32) == sub_16982C0() )
    {
      v19 = *(_QWORD *)(v10 + 40);
      if ( (*(_BYTE *)(v19 + 26) & 7) != 3 )
        return 0;
      v11 = v19 + 8;
    }
    else
    {
      if ( (*(_BYTE *)(v10 + 50) & 7) != 3 )
        return 0;
      v11 = v10 + 32;
    }
    return ((*(_BYTE *)(v11 + 18) >> 3) ^ 1) & 1;
  }
  return 0;
}
