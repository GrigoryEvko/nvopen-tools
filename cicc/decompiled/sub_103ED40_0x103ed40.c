// Function: sub_103ED40
// Address: 0x103ed40
//
_BOOL8 __fastcall sub_103ED40(__int64 a1, __int64 a2, __int64 *a3)
{
  int v3; // r12d
  _BOOL4 v4; // r8d
  __int64 v6; // rax
  bool v7; // zf
  __int64 v8; // r10
  int v9; // r13d
  int v10; // r12d
  __int64 v11; // r14
  unsigned int j; // r13d
  __int64 v13; // r15
  unsigned __int64 v14; // rax
  unsigned __int8 *v15; // rcx
  int v16; // edx
  __int64 v17; // r14
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r15
  __int64 v21; // rax
  int v22; // r15d
  __int64 v23; // rax
  __int64 v24; // rdx
  unsigned __int8 *i; // r14
  __int64 v26; // rax
  unsigned int v27; // eax
  bool v28; // al
  unsigned int v29; // r13d
  int v30; // [rsp+4h] [rbp-DCh]
  __int64 v31; // [rsp+8h] [rbp-D8h]
  __int64 v33; // [rsp+18h] [rbp-C8h]
  unsigned __int8 *v34; // [rsp+18h] [rbp-C8h]
  unsigned __int8 *v35; // [rsp+18h] [rbp-C8h]
  unsigned int v36; // [rsp+24h] [rbp-BCh] BYREF
  unsigned __int64 v37; // [rsp+28h] [rbp-B8h] BYREF
  _QWORD v38[8]; // [rsp+30h] [rbp-B0h] BYREF
  _QWORD v39[14]; // [rsp+70h] [rbp-70h] BYREF

  v3 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = *(_BYTE *)a2 == 0;
  v38[0] = 0;
  v38[1] = -4096;
  v8 = *(_QWORD *)(a2 + 8);
  v31 = v6;
  v38[2] = -3;
  memset(&v38[3], 0, 32);
  v39[0] = 0;
  v39[1] = -8192;
  v39[2] = -4;
  memset(&v39[3], 0, 32);
  if ( v7 )
  {
    LODWORD(v37) = ((0xBF58476D1CE4E5B9LL * *(_QWORD *)(a2 + 16)) >> 31)
                 ^ (484763065 * *(_DWORD *)(a2 + 16))
                 ^ ((unsigned int)*(_QWORD *)(a2 + 48) >> 9)
                 ^ ((unsigned int)*(_QWORD *)(a2 + 48) >> 4)
                 ^ ((unsigned int)*(_QWORD *)(a2 + 40) >> 9)
                 ^ ((unsigned int)*(_QWORD *)(a2 + 40) >> 4)
                 ^ ((unsigned int)*(_QWORD *)(a2 + 32) >> 9)
                 ^ ((unsigned int)*(_QWORD *)(a2 + 32) >> 4)
                 ^ ((unsigned int)*(_QWORD *)(a2 + 24) >> 4)
                 ^ ((unsigned int)v8 >> 9)
                 ^ ((unsigned int)v8 >> 4)
                 ^ ((unsigned int)*(_QWORD *)(a2 + 24) >> 9);
    v9 = sub_103EC40((char *)a2, (int *)&v37);
    goto LABEL_6;
  }
  v36 = ((unsigned int)*(_QWORD *)(v8 - 32) >> 4) ^ ((unsigned int)*(_QWORD *)(v8 - 32) >> 9);
  v14 = sub_103EC40((char *)a2, (int *)&v36);
  v15 = *(unsigned __int8 **)(a2 + 8);
  v37 = v14;
  v16 = *v15;
  if ( v16 == 40 )
  {
    v35 = v15;
    v27 = sub_B491D0((__int64)v15);
    v15 = v35;
    v17 = -32 - 32LL * v27;
  }
  else
  {
    v17 = -32;
    if ( v16 != 85 )
    {
      v17 = -96;
      if ( v16 != 34 )
        BUG();
    }
  }
  if ( (v15[7] & 0x80u) != 0 )
  {
    v33 = (__int64)v15;
    v18 = sub_BD2BC0((__int64)v15);
    v15 = (unsigned __int8 *)v33;
    v20 = v18 + v19;
    if ( *(char *)(v33 + 7) >= 0 )
    {
      if ( (unsigned int)(v20 >> 4) )
        goto LABEL_34;
    }
    else
    {
      v21 = sub_BD2BC0(v33);
      v15 = (unsigned __int8 *)v33;
      if ( (unsigned int)((v20 - v21) >> 4) )
      {
        if ( *(char *)(v33 + 7) < 0 )
        {
          v22 = *(_DWORD *)(sub_BD2BC0(v33) + 8);
          if ( *(char *)(v33 + 7) >= 0 )
            BUG();
          v23 = sub_BD2BC0(v33);
          v15 = (unsigned __int8 *)v33;
          v17 -= 32LL * (unsigned int)(*(_DWORD *)(v23 + v24 - 4) - v22);
          goto LABEL_17;
        }
LABEL_34:
        BUG();
      }
    }
  }
LABEL_17:
  v34 = &v15[v17];
  for ( i = &v15[-32 * (*((_DWORD *)v15 + 1) & 0x7FFFFFF)]; v34 != i; v37 = sub_103ECC0((__int64 *)&v37, &v36) )
  {
    v26 = *(_QWORD *)i;
    i += 32;
    v36 = ((unsigned int)v26 >> 4) ^ ((unsigned int)v26 >> 9);
  }
  v9 = v37;
LABEL_6:
  v10 = v3 - 1;
  v11 = 0;
  v30 = 1;
  for ( j = v10 & v9; ; j = v10 & v29 )
  {
    v13 = v31 + 104LL * j;
    v4 = sub_103B1D0(a2, v13);
    if ( v4 )
    {
      *a3 = v13;
      return v4;
    }
    if ( sub_103B1D0(v13, (__int64)v38) )
      break;
    v28 = sub_103B1D0(v13, (__int64)v39);
    if ( !v11 && v28 )
      v11 = v31 + 104LL * j;
    v29 = v30 + j;
    ++v30;
  }
  v4 = 0;
  if ( !v11 )
    v11 = v31 + 104LL * j;
  *a3 = v11;
  return v4;
}
