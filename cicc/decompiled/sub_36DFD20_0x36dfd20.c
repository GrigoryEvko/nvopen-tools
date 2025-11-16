// Function: sub_36DFD20
// Address: 0x36dfd20
//
__int64 __fastcall sub_36DFD20(__int64 a1, __int64 a2, int a3)
{
  int v3; // eax
  unsigned int v5; // r8d
  __int64 v6; // rax
  __int64 v7; // rcx
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  unsigned int v10; // r8d

  v3 = *(_DWORD *)(a2 + 24);
  if ( v3 > 365 )
  {
    if ( v3 > 470 )
    {
      if ( v3 == 497 )
      {
        v5 = 0;
        v6 = **(_QWORD **)(a2 + 112);
        if ( a3 )
          goto LABEL_9;
        goto LABEL_21;
      }
    }
    else if ( v3 > 464 )
    {
      v5 = 0;
      v6 = **(_QWORD **)(a2 + 112);
      if ( a3 )
        goto LABEL_9;
      goto LABEL_21;
    }
  }
  else
  {
    if ( v3 > 337 )
      goto LABEL_8;
    if ( v3 > 294 )
    {
      if ( (unsigned int)(v3 - 298) <= 1 )
        goto LABEL_8;
    }
    else if ( v3 > 292 )
    {
      goto LABEL_8;
    }
  }
  if ( (*(_BYTE *)(a2 + 32) & 2) == 0 )
    return 0;
LABEL_8:
  v5 = 0;
  v6 = **(_QWORD **)(a2 + 112);
  if ( a3 )
  {
LABEL_9:
    if ( !v6 )
      return 0;
    v7 = v6 >> 2;
    goto LABEL_11;
  }
LABEL_21:
  if ( !v6 )
    return 0;
  v7 = v6 >> 2;
  if ( (v6 & 4) != 0 )
  {
    LOBYTE(v5) = (v6 & 0xFFFFFFFFFFFFFFF8LL) != 0;
    return v5;
  }
LABEL_11:
  if ( (v7 & 1) != 0 )
    return 0;
  v8 = v6 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v8 )
    return 0;
  v9 = *(_QWORD *)(v8 + 8);
  v10 = 0;
  if ( *(_BYTE *)(v9 + 8) != 14 )
    return 0;
  LOBYTE(v10) = *(_DWORD *)(v9 + 8) >> 8 == a3;
  return v10;
}
