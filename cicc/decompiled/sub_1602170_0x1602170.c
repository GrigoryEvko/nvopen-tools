// Function: sub_1602170
// Address: 0x1602170
//
__int64 __fastcall sub_1602170(__int64 a1)
{
  __int64 v1; // rax
  int v2; // r12d
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r13
  int v6; // r13d
  __int64 v7; // rax
  __int64 v8; // rdx
  int v9; // ecx
  __int64 v10; // rax
  _BYTE *v11; // rdi
  unsigned int v12; // r12d
  __int64 v13; // rax
  __int64 v14; // rdx

  v1 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  v2 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( *(char *)(a1 + 23) >= 0 )
  {
    v9 = 0;
    goto LABEL_7;
  }
  v3 = sub_1648A40(a1);
  v5 = v3 + v4;
  if ( *(char *)(a1 + 23) >= 0 )
  {
    if ( (unsigned int)(v5 >> 4) )
LABEL_30:
      BUG();
LABEL_15:
    v9 = 0;
    v1 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
    goto LABEL_7;
  }
  if ( !(unsigned int)((v5 - sub_1648A40(a1)) >> 4) )
    goto LABEL_15;
  if ( *(char *)(a1 + 23) >= 0 )
    goto LABEL_30;
  v6 = *(_DWORD *)(sub_1648A40(a1) + 8);
  if ( *(char *)(a1 + 23) >= 0 )
    BUG();
  v7 = sub_1648A40(a1);
  v9 = *(_DWORD *)(v7 + v8 - 4) - v6;
  v1 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
LABEL_7:
  v10 = *(_QWORD *)(a1 + 24 * ((unsigned int)(v2 - 2 - v9) - v1));
  if ( *(_BYTE *)(v10 + 16) != 19 )
    BUG();
  v11 = *(_BYTE **)(v10 + 24);
  v12 = 0;
  if ( v11 && !*v11 )
  {
    v13 = sub_161E970(v11);
    if ( v14 == 15 )
    {
      if ( *(_QWORD *)v13 != 0x7470656378657066LL
        || *(_DWORD *)(v13 + 8) != 1852270894
        || *(_WORD *)(v13 + 12) != 29295
        || (v12 = 1, *(_BYTE *)(v13 + 14) != 101) )
      {
        if ( *(_QWORD *)v13 != 0x7470656378657066LL )
          return 0;
        if ( *(_DWORD *)(v13 + 8) != 1920234286 )
          return 0;
        if ( *(_WORD *)(v13 + 12) != 25449 )
          return 0;
        v12 = 3;
        if ( *(_BYTE *)(v13 + 14) != 116 )
          return 0;
      }
    }
    else if ( v14 == 16 )
    {
      return 2
           * (unsigned int)((*(_QWORD *)v13 ^ 0x7470656378657066LL | *(_QWORD *)(v13 + 8) ^ 0x7061727479616D2ELL) == 0);
    }
  }
  return v12;
}
