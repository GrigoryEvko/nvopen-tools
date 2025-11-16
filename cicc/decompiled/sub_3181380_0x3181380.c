// Function: sub_3181380
// Address: 0x3181380
//
__int64 __fastcall sub_3181380(__int64 *a1, __int64 a2, __int64 *a3, int a4)
{
  int v5; // edx
  __int64 *v6; // rbx
  __int64 v7; // rax
  __int64 *v8; // r12
  __int64 v9; // r15
  unsigned int v10; // r15d
  char v12; // cl
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 *v18; // rbx
  __int64 v19; // r12
  unsigned int v20; // eax
  unsigned __int8 *i; // rdi
  int v22; // edi
  unsigned __int8 *v23; // rax
  __int64 v24; // r12
  unsigned __int8 v25; // al
  __int64 v26; // r8
  __int64 v27; // [rsp+8h] [rbp-38h]
  int v28; // [rsp+8h] [rbp-38h]
  __int64 *v29; // [rsp+8h] [rbp-38h]

  if ( a4 == 22 )
    return 0;
  v5 = *(unsigned __int8 *)a1;
  v6 = a1;
  if ( (_BYTE)v5 == 82 )
  {
    if ( sub_3108430(*(a1 - 4), *a3) )
    {
LABEL_4:
      v7 = 32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
      if ( (*((_BYTE *)a1 + 7) & 0x40) != 0 )
      {
        v8 = (__int64 *)*(a1 - 1);
        v6 = &v8[(unsigned __int64)v7 / 8];
      }
      else
      {
        v8 = &a1[v7 / 0xFFFFFFFFFFFFFFF8LL];
      }
      while ( v6 != v8 )
      {
        v9 = *v8;
        if ( sub_3108430(*v8, *a3) )
        {
          v10 = sub_31843D0(a3, a2, v9);
          if ( (_BYTE)v10 )
            return v10;
        }
        v8 += 4;
      }
    }
    return 0;
  }
  v12 = v5 - 34;
  if ( (unsigned __int8)(v5 - 34) > 0x33u )
    goto LABEL_4;
  v10 = ((0x8000000000041uLL >> v12) & 1) == 0;
  if ( ((0x8000000000041uLL >> v12) & 1) != 0 )
  {
    if ( v5 == 40 )
    {
      v13 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a1);
    }
    else
    {
      v13 = -32;
      if ( v5 != 85 )
      {
        v13 = -96;
        if ( v5 != 34 )
LABEL_49:
          BUG();
      }
    }
    if ( *((char *)a1 + 7) < 0 )
    {
      v14 = sub_BD2BC0((__int64)a1);
      v27 = v15 + v14;
      if ( *((char *)a1 + 7) >= 0 )
      {
        if ( (unsigned int)(v27 >> 4) )
          goto LABEL_49;
      }
      else if ( (unsigned int)((v27 - sub_BD2BC0((__int64)a1)) >> 4) )
      {
        if ( *((char *)a1 + 7) >= 0 )
          goto LABEL_49;
        v28 = *(_DWORD *)(sub_BD2BC0((__int64)a1) + 8);
        if ( *((char *)a1 + 7) >= 0 )
          BUG();
        v16 = sub_BD2BC0((__int64)a1);
        v13 -= 32LL * (unsigned int)(*(_DWORD *)(v16 + v17 - 4) - v28);
      }
    }
    v29 = (__int64 *)((char *)a1 + v13);
    v18 = &a1[-4 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
    if ( v18 != (__int64 *)((char *)a1 + v13) )
    {
      while ( 1 )
      {
        v19 = *v18;
        if ( sub_3108430(*v18, *a3) )
        {
          v20 = sub_31843D0(a3, a2, v19);
          if ( (_BYTE)v20 )
            break;
        }
        v18 += 4;
        if ( v29 == v18 )
          return v10;
      }
      return v20;
    }
    return v10;
  }
  if ( (_BYTE)v5 != 62 )
    goto LABEL_4;
  for ( i = (unsigned __int8 *)*(a1 - 4); ; i = *(unsigned __int8 **)(v24 - 32LL * (*(_DWORD *)(v24 + 4) & 0x7FFFFFF)) )
  {
    v23 = sub_98ACB0(i, 6u);
    v22 = 23;
    v24 = (__int64)v23;
    v25 = *v23;
    if ( v25 > 0x1Cu )
    {
      if ( v25 == 85 )
      {
        v26 = *(_QWORD *)(v24 - 32);
        v22 = 21;
        if ( v26 && !*(_BYTE *)v26 && *(_QWORD *)(v26 + 24) == *(_QWORD *)(v24 + 80) )
          v22 = sub_3108960(*(_QWORD *)(v24 - 32));
      }
      else
      {
        v22 = 2 * (v25 != 34) + 21;
      }
    }
    if ( !(unsigned __int8)sub_3108CA0(v22) )
      break;
  }
  if ( !sub_3108430(v24, *a3) )
    return 0;
  return sub_31843D0(a3, v24, a2);
}
