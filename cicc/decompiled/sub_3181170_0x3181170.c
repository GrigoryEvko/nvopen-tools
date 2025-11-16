// Function: sub_3181170
// Address: 0x3181170
//
__int64 __fastcall sub_3181170(unsigned __int8 *a1, __int64 a2, __int64 *a3, unsigned int a4)
{
  unsigned int v5; // edx
  __int64 result; // rax
  int v7; // edx
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r15
  int v12; // r15d
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 *v15; // r14
  __int64 *v16; // rbx
  __int64 v17; // r15

  if ( a4 <= 0x17 && ((1LL << a4) & 0x900060) != 0 )
    return 0;
  v5 = sub_CF5CA0(*a3, (__int64)a1);
  if ( (((unsigned __int8)(v5 >> 6) | (unsigned __int8)((v5 >> 4) | v5 | (v5 >> 2))) & 2) == 0 )
    return 0;
  result = 1;
  if ( (v5 & 0xFFFFFFFC) != 0 )
    return result;
  v7 = *a1;
  if ( v7 == 40 )
  {
    v8 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a1);
  }
  else
  {
    v8 = -32;
    if ( v7 != 85 )
    {
      v8 = -96;
      if ( v7 != 34 )
        BUG();
    }
  }
  if ( (a1[7] & 0x80u) == 0 )
    goto LABEL_15;
  v9 = sub_BD2BC0((__int64)a1);
  v11 = v9 + v10;
  if ( (a1[7] & 0x80u) == 0 )
  {
    if ( !(unsigned int)(v11 >> 4) )
      goto LABEL_15;
LABEL_25:
    BUG();
  }
  if ( !(unsigned int)((v11 - sub_BD2BC0((__int64)a1)) >> 4) )
    goto LABEL_15;
  if ( (a1[7] & 0x80u) == 0 )
    goto LABEL_25;
  v12 = *(_DWORD *)(sub_BD2BC0((__int64)a1) + 8);
  if ( (a1[7] & 0x80u) == 0 )
    BUG();
  v13 = sub_BD2BC0((__int64)a1);
  v8 -= 32LL * (unsigned int)(*(_DWORD *)(v13 + v14 - 4) - v12);
LABEL_15:
  v15 = (__int64 *)&a1[v8];
  v16 = (__int64 *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
  if ( v16 == v15 )
    return 0;
  while ( 1 )
  {
    v17 = *v16;
    if ( sub_3108430(*v16, *a3) )
    {
      result = sub_31843D0(a3, a2, v17);
      if ( (_BYTE)result )
        break;
    }
    v16 += 4;
    if ( v15 == v16 )
      return 0;
  }
  return result;
}
