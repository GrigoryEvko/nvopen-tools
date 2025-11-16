// Function: sub_173D590
// Address: 0x173d590
//
__int64 __fastcall sub_173D590(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r12d
  unsigned __int8 v5; // al
  __int64 v6; // rbx
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // r13
  unsigned int v11; // r15d
  int v12; // r14d
  __int64 v13; // rax
  __int64 v14; // r13
  char v15; // al
  __int64 v16; // r13

  v5 = a1[16];
  if ( v5 == 14 )
  {
    if ( *((void **)a1 + 4) == sub_16982C0() )
      v6 = *((_QWORD *)a1 + 5) + 8LL;
    else
      v6 = (__int64)(a1 + 32);
    LOBYTE(v4) = (*(_BYTE *)(v6 + 18) & 7) == 1;
    return v4;
  }
  LOBYTE(v4) = v5 <= 0x10u && *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 16;
  if ( !(_BYTE)v4 )
    return 0;
  v8 = sub_15A1020(a1, a2, *(_QWORD *)a1, a4);
  v9 = v8;
  if ( !v8 || *(_BYTE *)(v8 + 16) != 14 )
  {
    v11 = 0;
    v12 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
    if ( !v12 )
      return v4;
    while ( 1 )
    {
      v13 = sub_15A0A60((__int64)a1, v11);
      v14 = v13;
      if ( !v13 )
        break;
      v15 = *(_BYTE *)(v13 + 16);
      if ( v15 != 9 )
      {
        if ( v15 != 14 )
          break;
        v16 = *(void **)(v14 + 32) == sub_16982C0() ? *(_QWORD *)(v14 + 40) + 8LL : v14 + 32;
        if ( (*(_BYTE *)(v16 + 18) & 7) != 1 )
          break;
      }
      if ( v12 == ++v11 )
        return v4;
    }
    return 0;
  }
  if ( *(void **)(v8 + 32) == sub_16982C0() )
    v10 = *(_QWORD *)(v9 + 40) + 8LL;
  else
    v10 = v9 + 32;
  LOBYTE(v4) = (*(_BYTE *)(v10 + 18) & 7) == 1;
  return v4;
}
