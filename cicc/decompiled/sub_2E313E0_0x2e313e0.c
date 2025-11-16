// Function: sub_2E313E0
// Address: 0x2e313e0
//
unsigned __int64 __fastcall sub_2E313E0(__int64 a1)
{
  __int64 v1; // rbx
  unsigned __int64 v2; // r12
  __int64 v3; // r13
  __int64 v4; // rax
  int v5; // edx
  __int64 v6; // rax
  int v7; // eax

  v1 = a1 + 48;
  v2 = a1 + 48;
  v3 = *(_QWORD *)(a1 + 56);
  if ( v3 == a1 + 48 )
    return v2;
  do
  {
    v2 = *(_QWORD *)v2 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v2 )
      BUG();
    v4 = *(_QWORD *)v2;
    v5 = *(_DWORD *)(v2 + 44);
    if ( (*(_QWORD *)v2 & 4) != 0 )
    {
      if ( (v5 & 4) != 0 )
      {
LABEL_25:
        v6 = (*(_QWORD *)(*(_QWORD *)(v2 + 16) + 24LL) >> 9) & 1LL;
        goto LABEL_10;
      }
    }
    else if ( (v5 & 4) != 0 )
    {
      while ( 1 )
      {
        v2 = v4 & 0xFFFFFFFFFFFFFFF8LL;
        LOBYTE(v5) = *(_DWORD *)((v4 & 0xFFFFFFFFFFFFFFF8LL) + 44);
        if ( (v5 & 4) == 0 )
          break;
        v4 = *(_QWORD *)v2;
      }
    }
    if ( (v5 & 8) == 0 )
      goto LABEL_25;
    LOBYTE(v6) = sub_2E88A90(v2, 512, 1);
LABEL_10:
    if ( !(_BYTE)v6 && (unsigned __int16)(*(_WORD *)(v2 + 68) - 14) > 4u )
    {
      if ( v1 != v2 )
        goto LABEL_18;
      return v2;
    }
  }
  while ( v3 != v2 );
  v7 = *(_DWORD *)(v2 + 44);
  if ( (v7 & 4) != 0 )
    goto LABEL_19;
LABEL_14:
  if ( (v7 & 8) != 0 )
  {
    if ( !(unsigned __int8)sub_2E88A90(v2, 512, 1) )
      goto LABEL_16;
  }
  else
  {
LABEL_19:
    while ( (*(_QWORD *)(*(_QWORD *)(v2 + 16) + 24LL) & 0x200LL) == 0 )
    {
LABEL_16:
      if ( (*(_BYTE *)v2 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v2 + 44) & 8) != 0 )
          v2 = *(_QWORD *)(v2 + 8);
      }
      v2 = *(_QWORD *)(v2 + 8);
      if ( v1 == v2 )
        return v2;
LABEL_18:
      v7 = *(_DWORD *)(v2 + 44);
      if ( (v7 & 4) == 0 )
        goto LABEL_14;
    }
  }
  return v2;
}
