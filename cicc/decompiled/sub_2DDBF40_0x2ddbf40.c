// Function: sub_2DDBF40
// Address: 0x2ddbf40
//
__int64 __fastcall sub_2DDBF40(unsigned __int8 *a1, unsigned int a2)
{
  __int64 v2; // rax
  int v4; // ecx
  unsigned __int64 v5; // rdx
  _BOOL4 v7; // r14d
  __int64 v8; // rsi
  unsigned __int8 *v10; // rdx
  __int64 v11; // rbx
  unsigned __int8 v12; // cl
  unsigned __int8 *v13; // rdi
  unsigned __int8 *v14; // rax
  const char *v15; // rax
  unsigned __int64 v16; // rdx
  char v17; // dl
  unsigned __int8 *v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rbx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r13
  __int64 v27; // r13
  __int64 v28; // rbx
  __int64 v29; // r13
  __int64 v30; // rax

  v2 = *((_DWORD *)a1 + 1) & 0x7FFFFFF;
  if ( a2 >= (unsigned int)v2 )
    return 0;
  v4 = *a1;
  v5 = (unsigned int)(v4 - 29);
  if ( (unsigned int)v5 > 0x38 )
    return 0;
  v7 = 0;
  v8 = 0x100000300000020LL;
  if ( !_bittest64(&v8, v5) )
    return v7;
  v10 = (a1[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)a1 - 1) : &a1[-32 * v2];
  v11 = 32LL * a2;
  if ( **(_BYTE **)&v10[v11] > 0x15u )
    return 0;
  v12 = v4 - 34;
  if ( v12 > 0x33u )
    return 1;
  v7 = ((0x8000000000041uLL >> v12) & 1) == 0;
  if ( ((0x8000000000041uLL >> v12) & 1) == 0 )
    return 1;
  v13 = (unsigned __int8 *)*((_QWORD *)a1 - 4);
  if ( *v13 == 25 )
    return v7;
  v14 = sub_BD3990(v13, 0x100000300000020LL);
  if ( v14 && !*v14 )
  {
    if ( (v14[33] & 0x20) != 0 )
      return v7;
    v15 = sub_BD5D20((__int64)v14);
    if ( v16 <= 0xC )
    {
      if ( v16 <= 7 )
        goto LABEL_18;
    }
    else if ( *(_QWORD *)v15 == 0x67736D5F636A626FLL && *((_DWORD *)v15 + 2) == 1684956499 && v15[12] == 36 )
    {
      return v7;
    }
    if ( *(_QWORD *)v15 == 0x6563617274645F5FLL )
      return v7;
  }
LABEL_18:
  v17 = a1[7];
  if ( (v17 & 0x40) != 0 )
    v18 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
  else
    v18 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
  if ( a1 - 32 == &v18[v11] )
  {
    if ( v17 < 0 )
    {
      v24 = sub_BD2BC0((__int64)a1);
      v26 = v24 + v25;
      if ( (a1[7] & 0x80u) != 0 )
        v26 -= sub_BD2BC0((__int64)a1);
      v27 = v26 >> 4;
      if ( (_DWORD)v27 )
      {
        v28 = 0;
        v29 = 16LL * (unsigned int)v27;
        while ( 1 )
        {
          v30 = 0;
          if ( (a1[7] & 0x80u) != 0 )
            v30 = sub_BD2BC0((__int64)a1);
          if ( *(_DWORD *)(*(_QWORD *)(v30 + v28) + 8LL) == 7 )
            break;
          v28 += 16;
          if ( v29 == v28 )
            return 1;
        }
        return v7;
      }
    }
  }
  else if ( v17 < 0 )
  {
    v19 = sub_BD2BC0((__int64)a1);
    v21 = v19 + v20;
    if ( (a1[7] & 0x80u) == 0 )
    {
      if ( (unsigned int)(v21 >> 4) )
        goto LABEL_45;
    }
    else if ( (unsigned int)((v21 - sub_BD2BC0((__int64)a1)) >> 4) )
    {
      if ( (a1[7] & 0x80u) != 0 )
      {
        if ( a2 >= *(_DWORD *)(sub_BD2BC0((__int64)a1) + 8) )
        {
          if ( (a1[7] & 0x80u) == 0 )
            BUG();
          v22 = sub_BD2BC0((__int64)a1);
          if ( a2 < *(_DWORD *)(v22 + v23 - 4) )
          {
            LOBYTE(v7) = *(_DWORD *)(*(_QWORD *)sub_B49810((__int64)a1, a2) + 8LL) != 6;
            return v7;
          }
        }
        return 1;
      }
LABEL_45:
      BUG();
    }
  }
  return 1;
}
