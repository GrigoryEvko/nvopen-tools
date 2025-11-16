// Function: sub_2B11690
// Address: 0x2b11690
//
bool __fastcall sub_2B11690(__int64 a1, unsigned __int8 *a2, __int64 *a3, __int64 a4)
{
  unsigned int v4; // eax
  unsigned int v7; // eax
  int v8; // edx
  unsigned int v9; // r15d
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned __int8 *v15; // rbx
  unsigned int v16; // r14d
  __int64 v17; // [rsp-40h] [rbp-40h]
  int v18; // [rsp-40h] [rbp-40h]
  unsigned __int8 *v19; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return 0;
  v4 = *a2 - 29;
  if ( v4 <= 0x21 )
  {
    if ( v4 > 0x1F )
      return *((_QWORD *)a2 - 4) == a1;
    return 0;
  }
  if ( *a2 != 85 )
    return 0;
  v7 = sub_9B78C0((__int64)a2, a3);
  v8 = *a2;
  v9 = v7;
  if ( v8 == 40 )
  {
    v10 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v10 = -32;
    if ( v8 != 85 )
    {
      v10 = -96;
      if ( v8 != 34 )
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) != 0 )
  {
    v11 = sub_BD2BC0((__int64)a2);
    v17 = v12 + v11;
    if ( (a2[7] & 0x80u) == 0 )
    {
      if ( (unsigned int)(v17 >> 4) )
        goto LABEL_28;
    }
    else if ( (unsigned int)((v17 - sub_BD2BC0((__int64)a2)) >> 4) )
    {
      if ( (a2[7] & 0x80u) != 0 )
      {
        v18 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
        if ( (a2[7] & 0x80u) == 0 )
          BUG();
        v13 = sub_BD2BC0((__int64)a2);
        v10 -= 32LL * (unsigned int)(*(_DWORD *)(v13 + v14 - 4) - v18);
        goto LABEL_15;
      }
LABEL_28:
      BUG();
    }
  }
LABEL_15:
  v19 = &a2[v10];
  v15 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  if ( v15 != &a2[v10] )
  {
    v16 = 0;
    do
    {
      if ( sub_9B75A0(v9, v16, a4) && a1 == *(_QWORD *)v15 )
        break;
      v15 += 32;
      ++v16;
    }
    while ( v19 != v15 );
  }
  return v19 != v15;
}
