// Function: sub_139C570
// Address: 0x139c570
//
char __fastcall sub_139C570(__int64 a1, unsigned int a2, __int64 a3)
{
  unsigned int v3; // r13d
  __int64 v4; // rax
  char result; // al
  int v6; // r15d
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r12
  int v10; // r12d
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned int v13; // r12d
  __int64 v14; // rax
  __int64 v15; // rdx
  _DWORD *v16; // r14
  _DWORD *v17; // rax
  unsigned int v18; // edx
  __int64 v19; // rax
  _QWORD v20[7]; // [rsp+8h] [rbp-38h] BYREF

  v3 = a3;
  if ( !a2 )
  {
    if ( !(unsigned __int8)sub_1560260(a1 + 56, 0, a3) )
    {
      v4 = *(_QWORD *)(a1 - 72);
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
  if ( *(char *)(a1 + 23) >= 0 )
    goto LABEL_23;
  v7 = sub_1648A40(a1);
  v9 = v7 + v8;
  if ( *(char *)(a1 + 23) >= 0 )
  {
    if ( (unsigned int)(v9 >> 4) )
LABEL_31:
      BUG();
LABEL_23:
    v13 = a2 - 1;
    if ( v6 - 2 <= a2 )
      goto LABEL_12;
LABEL_24:
    if ( !(unsigned __int8)sub_1560290(a1 + 56, v13, v3) )
    {
      v19 = *(_QWORD *)(a1 - 72);
      if ( !*(_BYTE *)(v19 + 16) )
      {
        v20[0] = *(_QWORD *)(v19 + 112);
        return sub_1560290(v20, v13, v3);
      }
      return 0;
    }
    return 1;
  }
  if ( !(unsigned int)((v9 - sub_1648A40(a1)) >> 4) )
    goto LABEL_23;
  if ( *(char *)(a1 + 23) >= 0 )
    goto LABEL_31;
  v10 = *(_DWORD *)(sub_1648A40(a1) + 8);
  if ( *(char *)(a1 + 23) >= 0 )
    BUG();
  v11 = sub_1648A40(a1);
  LODWORD(v11) = *(_DWORD *)(v11 + v12 - 4) - v10;
  v13 = a2 - 1;
  if ( v6 - 2 - (int)v11 > a2 )
    goto LABEL_24;
LABEL_12:
  if ( *(char *)(a1 + 23) >= 0 )
    goto LABEL_31;
  v14 = sub_1648A40(a1);
  v16 = (_DWORD *)(v14 + v15);
  if ( *(char *)(a1 + 23) >= 0 )
    v17 = 0;
  else
    v17 = (_DWORD *)sub_1648A40(a1);
  while ( 1 )
  {
    if ( v17 == v16 )
      goto LABEL_31;
    v18 = v17[2];
    if ( v18 <= v13 && v17[3] > v13 )
      break;
    v17 += 4;
  }
  if ( *(_DWORD *)(*(_QWORD *)v17 + 8LL) )
    return 0;
  result = v3 == 22 || v3 == 37;
  if ( result )
    return *(_BYTE *)(**(_QWORD **)(a1 + 24 * (v18 - (unsigned __int64)(*(_DWORD *)(a1 + 20) & 0xFFFFFFF) + v13 - v18))
                    + 8LL) == 15;
  return result;
}
