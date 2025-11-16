// Function: sub_17917B0
// Address: 0x17917b0
//
__int64 __fastcall sub_17917B0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // r13
  __int64 v4; // r12
  _QWORD *v5; // rax
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rcx
  __int64 v12; // rcx
  unsigned __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rcx
  unsigned __int64 v17; // rdx
  __int64 v18; // rdx

  v2 = *(_QWORD *)(a1 + 8);
  v3 = *(_QWORD *)(a1 - 72);
  if ( v2 )
  {
    v4 = *(_QWORD *)(v2 + 8);
    if ( !v4 )
    {
      v5 = sub_1648700(v2);
      if ( *((_BYTE *)v5 + 16) == 79
        && *(v5 - 9) == v3
        && (*(v5 - 3) == *(_QWORD *)(a1 - 48) || *(v5 - 6) == *(_QWORD *)(a1 - 24)) )
      {
        return v4;
      }
    }
  }
  v4 = 0;
  if ( *(_BYTE *)(v3 + 16) != 86 || **(_DWORD **)(v3 + 56) != 1 )
    return v4;
  v7 = *(_QWORD *)(v3 - 24);
  if ( !v7 )
    goto LABEL_40;
  if ( *(_BYTE *)(v7 + 16) != 58 )
    return v4;
  v8 = *(_QWORD *)(a1 - 48);
  if ( *(_BYTE *)(v8 + 16) != 86 || **(_DWORD **)(v8 + 56) )
  {
    v9 = *(_QWORD *)(a1 - 24);
  }
  else
  {
    v15 = *(_QWORD *)(v8 - 24);
    if ( !v15 )
LABEL_40:
      BUG();
    v9 = *(_QWORD *)(a1 - 24);
    if ( *(_BYTE *)(v15 + 16) == 58 && v7 == v15 && *(_QWORD *)(v15 - 48) == v9 )
    {
      v16 = *(_QWORD *)(a1 - 40);
      v17 = *(_QWORD *)(a1 - 32) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v17 = v16;
      if ( v16 )
        *(_QWORD *)(v16 + 16) = *(_QWORD *)(v16 + 16) & 3LL | v17;
      *(_QWORD *)(a1 - 48) = v9;
      if ( v9 )
      {
        v18 = *(_QWORD *)(v9 + 8);
        *(_QWORD *)(a1 - 40) = v18;
        if ( v18 )
          *(_QWORD *)(v18 + 16) = (a1 - 40) | *(_QWORD *)(v18 + 16) & 3LL;
        *(_QWORD *)(a1 - 32) = (v9 + 8) | *(_QWORD *)(a1 - 32) & 3LL;
        *(_QWORD *)(v9 + 8) = a1 - 48;
      }
      return a1;
    }
  }
  v4 = 0;
  if ( *(_BYTE *)(v9 + 16) != 86 || **(_DWORD **)(v9 + 56) )
    return v4;
  v10 = *(_QWORD *)(v9 - 24);
  if ( !v10 )
    goto LABEL_40;
  if ( *(_BYTE *)(v10 + 16) == 58 && v7 == v10 )
  {
    v11 = *(_QWORD *)(v10 - 48);
    if ( v8 == v11 )
    {
      if ( v11 )
      {
        v12 = *(_QWORD *)(a1 - 40);
        v13 = *(_QWORD *)(a1 - 32) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v13 = v12;
        if ( v12 )
          *(_QWORD *)(v12 + 16) = *(_QWORD *)(v12 + 16) & 3LL | v13;
        *(_QWORD *)(a1 - 48) = v9;
        v14 = *(_QWORD *)(v9 + 8);
        *(_QWORD *)(a1 - 40) = v14;
        if ( v14 )
          *(_QWORD *)(v14 + 16) = (a1 - 40) | *(_QWORD *)(v14 + 16) & 3LL;
        v4 = a1;
        *(_QWORD *)(a1 - 32) = (v9 + 8) | *(_QWORD *)(a1 - 32) & 3LL;
        *(_QWORD *)(v9 + 8) = a1 - 48;
        return v4;
      }
    }
  }
  return 0;
}
