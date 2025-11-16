// Function: sub_1AE5870
// Address: 0x1ae5870
//
char __fastcall sub_1AE5870(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  int v5; // r8d
  int v6; // r9d
  int v7; // r14d
  __int64 v8; // rdx
  __int64 v9; // r12
  int v10; // r12d
  __int64 v11; // rdx
  unsigned int v13[10]; // [rsp+8h] [rbp-28h] BYREF

  if ( (unsigned __int8)sub_1560260((_QWORD *)(a2 + 56), -1, 21) )
    goto LABEL_2;
  v3 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v3 + 16) )
    return v3;
  *(_QWORD *)v13 = *(_QWORD *)(v3 + 112);
  LOBYTE(v3) = sub_1560260(v13, -1, 21);
  if ( (_BYTE)v3 )
  {
LABEL_2:
    LOBYTE(v3) = sub_1560260((_QWORD *)(a2 + 56), -1, 5);
    if ( !(_BYTE)v3 )
    {
      v3 = *(_QWORD *)(a2 - 24);
      if ( *(_BYTE *)(v3 + 16) )
        return v3;
      *(_QWORD *)v13 = *(_QWORD *)(v3 + 112);
      LOBYTE(v3) = sub_1560260(v13, -1, 5);
      if ( !(_BYTE)v3 )
        return v3;
    }
    if ( *(_QWORD *)(a2 + 8) )
      return v3;
  }
  else if ( *(_QWORD *)(a2 + 8) )
  {
    return v3;
  }
  v4 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v4 + 16) )
    return v3;
  LOBYTE(v3) = sub_149CB50(**(_QWORD **)a1, v4, v13);
  if ( !(_BYTE)v3 )
    return v3;
  LODWORD(v3) = (int)*(unsigned __int8 *)(**(_QWORD **)a1 + (signed int)v13[0] / 4) >> (2 * (v13[0] & 3));
  if ( (v3 & 3) == 0 )
    return v3;
  v7 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( *(char *)(a2 + 23) < 0 )
  {
    v3 = sub_1648A40(a2);
    v9 = v3 + v8;
    if ( *(char *)(a2 + 23) >= 0 )
    {
      if ( (unsigned int)(v9 >> 4) )
        goto LABEL_29;
    }
    else
    {
      v3 = sub_1648A40(a2);
      if ( (unsigned int)((v9 - v3) >> 4) )
      {
        if ( *(char *)(a2 + 23) < 0 )
        {
          v10 = *(_DWORD *)(sub_1648A40(a2) + 8);
          if ( *(char *)(a2 + 23) >= 0 )
            BUG();
          v3 = sub_1648A40(a2);
          v7 += v10 - *(_DWORD *)(v3 + v11 - 4);
          goto LABEL_14;
        }
LABEL_29:
        BUG();
      }
    }
  }
LABEL_14:
  if ( v7 != 1 )
  {
    LOBYTE(v3) = *(_BYTE *)(**(_QWORD **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)) + 8LL) - 2;
    if ( (unsigned __int8)v3 <= 2u )
    {
      v3 = *(unsigned int *)(a1 + 24);
      if ( (unsigned int)v3 >= *(_DWORD *)(a1 + 28) )
      {
        sub_16CD150(a1 + 16, (const void *)(a1 + 32), 0, 8, v5, v6);
        v3 = *(unsigned int *)(a1 + 24);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8 * v3) = a2;
      ++*(_DWORD *)(a1 + 24);
    }
  }
  return v3;
}
