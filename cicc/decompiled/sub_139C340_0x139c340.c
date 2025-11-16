// Function: sub_139C340
// Address: 0x139c340
//
char __fastcall sub_139C340(__int64 a1, unsigned int a2, __int64 a3)
{
  unsigned int v3; // r13d
  __int64 v4; // rax
  char result; // al
  unsigned int v6; // r15d
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r12
  int v10; // r12d
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned int v13; // r12d
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  _DWORD *v17; // r14
  _DWORD *v18; // rax
  unsigned int v19; // edx
  _QWORD v20[7]; // [rsp+8h] [rbp-38h] BYREF

  v3 = a3;
  if ( !a2 )
  {
    if ( !(unsigned __int8)sub_1560260(a1 + 56, 0, a3) )
    {
      v4 = *(_QWORD *)(a1 - 24);
      if ( !*(_BYTE *)(v4 + 16) )
      {
        v20[0] = *(_QWORD *)(v4 + 112);
        return sub_1560260(v20, 0, v3);
      }
      return 0;
    }
    return 1;
  }
  v6 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( *(char *)(a1 + 23) < 0 )
  {
    v7 = sub_1648A40(a1);
    v9 = v7 + v8;
    if ( *(char *)(a1 + 23) >= 0 )
    {
      if ( !(unsigned int)(v9 >> 4) )
        goto LABEL_12;
    }
    else
    {
      if ( !(unsigned int)((v9 - sub_1648A40(a1)) >> 4) )
        goto LABEL_12;
      if ( *(char *)(a1 + 23) < 0 )
      {
        v10 = *(_DWORD *)(sub_1648A40(a1) + 8);
        if ( *(char *)(a1 + 23) >= 0 )
          BUG();
        v11 = sub_1648A40(a1);
        v6 += v10 - *(_DWORD *)(v11 + v12 - 4);
        goto LABEL_12;
      }
    }
    BUG();
  }
LABEL_12:
  v13 = a2 - 1;
  if ( a2 < v6 )
  {
    if ( !(unsigned __int8)sub_1560290(a1 + 56, v13, v3) )
    {
      v14 = *(_QWORD *)(a1 - 24);
      if ( !*(_BYTE *)(v14 + 16) )
      {
        v20[0] = *(_QWORD *)(v14 + 112);
        return sub_1560290(v20, v13, v3);
      }
      return 0;
    }
    return 1;
  }
  if ( *(char *)(a1 + 23) < 0 )
  {
    v15 = sub_1648A40(a1);
    v17 = (_DWORD *)(v15 + v16);
    if ( *(char *)(a1 + 23) >= 0 )
      v18 = 0;
    else
      v18 = (_DWORD *)sub_1648A40(a1);
    while ( v18 != v17 )
    {
      v19 = v18[2];
      if ( v19 <= v13 && v18[3] > v13 )
      {
        if ( *(_DWORD *)(*(_QWORD *)v18 + 8LL) )
          return 0;
        result = v3 == 22 || v3 == 37;
        if ( result )
          return *(_BYTE *)(**(_QWORD **)(a1
                                        + 24 * (v19 - (unsigned __int64)(*(_DWORD *)(a1 + 20) & 0xFFFFFFF) + v13 - v19))
                          + 8LL) == 15;
        return result;
      }
      v18 += 4;
    }
  }
  return sub_139C570();
}
