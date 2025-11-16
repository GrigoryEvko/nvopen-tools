// Function: sub_30747D0
// Address: 0x30747d0
//
__int64 __fastcall sub_30747D0(__int64 a1, unsigned __int8 *a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // r13
  unsigned int v5; // r12d
  __int64 v7; // r12
  int v8; // r14d
  _BYTE *v9; // rdi
  int v10; // edx
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r14
  int v15; // r14d
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned __int8 *i; // rbx
  __int64 v19; // rax
  char v20; // cl

  v2 = sub_B43CB0((__int64)a2);
  v3 = *((_QWORD *)a2 - 4);
  if ( !v3 )
    return 0;
  if ( *(_BYTE *)v3 )
    return 0;
  v4 = v2;
  if ( *((_QWORD *)a2 + 10) != *(_QWORD *)(v3 + 24) || sub_B2FC80(*((_QWORD *)a2 - 4)) )
    return 0;
  if ( !(unsigned __int8)sub_CE9220(v4) )
    goto LABEL_14;
  v7 = *(_QWORD *)(v3 + 16);
  if ( !v7 )
    goto LABEL_14;
  v8 = 0;
  do
  {
    v9 = *(_BYTE **)(v7 + 24);
    if ( *v9 == 85 )
      v8 += v4 == sub_B43CB0((__int64)v9);
    v7 = *(_QWORD *)(v7 + 8);
  }
  while ( v7 );
  v5 = 5000;
  if ( v8 != 1 )
LABEL_14:
    v5 = 0;
  v10 = *a2;
  if ( v10 == 40 )
  {
    v11 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v11 = -32;
    if ( v10 != 85 )
    {
      v11 = -96;
      if ( v10 != 34 )
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) != 0 )
  {
    v12 = sub_BD2BC0((__int64)a2);
    v14 = v12 + v13;
    if ( (a2[7] & 0x80u) == 0 )
    {
      if ( (unsigned int)(v14 >> 4) )
        goto LABEL_39;
    }
    else if ( (unsigned int)((v14 - sub_BD2BC0((__int64)a2)) >> 4) )
    {
      if ( (a2[7] & 0x80u) != 0 )
      {
        v15 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
        if ( (a2[7] & 0x80u) == 0 )
          BUG();
        v16 = sub_BD2BC0((__int64)a2);
        v11 -= 32LL * (unsigned int)(*(_DWORD *)(v16 + v17 - 4) - v15);
        goto LABEL_26;
      }
LABEL_39:
      BUG();
    }
  }
LABEL_26:
  for ( i = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)]; &a2[v11] != i; i += 32 )
  {
    if ( **(_BYTE **)i == 60 )
    {
      v19 = *(_QWORD *)(*(_QWORD *)i + 72LL);
      v20 = *(_BYTE *)(v19 + 8);
      if ( v20 == 16 )
      {
        if ( *(_QWORD *)(v19 + 32) > 1u )
          v5 += 500;
      }
      else if ( v20 == 15 )
      {
        v5 += 500;
      }
    }
  }
  return v5;
}
