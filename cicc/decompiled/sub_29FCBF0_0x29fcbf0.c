// Function: sub_29FCBF0
// Address: 0x29fcbf0
//
char __fastcall sub_29FCBF0(__int64 a1, unsigned __int8 *a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  __int64 v5; // r8
  __int64 v6; // r9
  int v7; // edx
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r14
  int v12; // r14d
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned int v16[9]; // [rsp+Ch] [rbp-24h] BYREF

  if ( (unsigned __int8)sub_A73ED0((_QWORD *)a2 + 9, 23) || (LOBYTE(v3) = sub_B49560((__int64)a2, 23), (_BYTE)v3) )
  {
    LOBYTE(v3) = sub_A73ED0((_QWORD *)a2 + 9, 4);
    if ( !(_BYTE)v3 )
    {
      LOBYTE(v3) = sub_B49560((__int64)a2, 4);
      if ( !(_BYTE)v3 )
        return v3;
    }
    if ( *((_QWORD *)a2 + 2) )
      return v3;
  }
  else if ( *((_QWORD *)a2 + 2) )
  {
    return v3;
  }
  v4 = *((_QWORD *)a2 - 4);
  if ( !v4 )
    return v3;
  if ( *(_BYTE *)v4 )
    return v3;
  v3 = *(_QWORD *)(v4 + 24);
  if ( *((_QWORD *)a2 + 10) != v3 )
    return v3;
  LOBYTE(v3) = sub_981210(**(_QWORD **)a1, v4, v16);
  if ( !(_BYTE)v3 )
    return v3;
  v3 = *(_QWORD *)(*(_QWORD *)a1 + 8 * ((unsigned __int64)v16[0] >> 6) + 8) & (1LL << SLOBYTE(v16[0]));
  if ( v3 )
    return v3;
  LODWORD(v3) = (int)*(unsigned __int8 *)(**(_QWORD **)a1 + (v16[0] >> 2)) >> (2 * (v16[0] & 3));
  if ( (v3 & 3) == 0 )
    return v3;
  v7 = *a2;
  if ( v7 == 40 )
  {
    v8 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a2);
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
  if ( (a2[7] & 0x80u) != 0 )
  {
    v9 = sub_BD2BC0((__int64)a2);
    v11 = v9 + v10;
    if ( (a2[7] & 0x80u) == 0 )
    {
      if ( (unsigned int)(v11 >> 4) )
        goto LABEL_35;
    }
    else if ( (unsigned int)((v11 - sub_BD2BC0((__int64)a2)) >> 4) )
    {
      if ( (a2[7] & 0x80u) != 0 )
      {
        v12 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
        if ( (a2[7] & 0x80u) == 0 )
          BUG();
        v13 = sub_BD2BC0((__int64)a2);
        v8 -= 32LL * (unsigned int)(*(_DWORD *)(v13 + v14 - 4) - v12);
        goto LABEL_27;
      }
LABEL_35:
      BUG();
    }
  }
LABEL_27:
  v3 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
  if ( 32 * v3 + v8 )
  {
    LOBYTE(v3) = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)&a2[-32 * v3] + 8LL) + 8LL) - 2;
    if ( (unsigned __int8)v3 <= 2u )
    {
      v3 = *(unsigned int *)(a1 + 24);
      if ( v3 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
      {
        sub_C8D5F0(a1 + 16, (const void *)(a1 + 32), v3 + 1, 8u, v5, v6);
        v3 = *(unsigned int *)(a1 + 24);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8 * v3) = a2;
      ++*(_DWORD *)(a1 + 24);
    }
  }
  return v3;
}
