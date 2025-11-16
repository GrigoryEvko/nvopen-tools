// Function: sub_CF5550
// Address: 0xcf5550
//
__int64 __fastcall sub_CF5550(_QWORD *a1, unsigned __int8 *a2, unsigned __int8 *a3, __int64 a4)
{
  _QWORD *v6; // r15
  unsigned int v7; // r12d
  unsigned int v9; // r15d
  unsigned int v10; // eax
  int v11; // edx
  unsigned __int8 *v12; // r8
  __int64 v13; // r9
  int v14; // edx
  unsigned __int8 *v15; // r15
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  char v22; // al
  char v23; // r9
  unsigned int v24; // eax
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // r15
  __int64 v28; // rax
  int v29; // r15d
  __int64 v30; // rax
  __int64 v31; // rdx
  unsigned __int8 *v32; // r9
  char v33; // al
  __int64 v34; // [rsp+8h] [rbp-88h]
  unsigned __int8 *v35; // [rsp+8h] [rbp-88h]
  unsigned __int8 *v36; // [rsp+10h] [rbp-80h]
  __int64 v37; // [rsp+18h] [rbp-78h]
  unsigned __int8 v38; // [rsp+18h] [rbp-78h]
  __int64 v39; // [rsp+18h] [rbp-78h]
  __int64 v40; // [rsp+18h] [rbp-78h]
  char v41; // [rsp+18h] [rbp-78h]
  _QWORD *v42; // [rsp+28h] [rbp-68h]
  __int64 v43; // [rsp+28h] [rbp-68h]
  __int64 v44; // [rsp+28h] [rbp-68h]
  int v45; // [rsp+28h] [rbp-68h]
  unsigned __int8 *v46; // [rsp+28h] [rbp-68h]
  unsigned __int8 *v47; // [rsp+28h] [rbp-68h]
  unsigned __int8 *v48; // [rsp+28h] [rbp-68h]
  unsigned __int8 v49; // [rsp+28h] [rbp-68h]
  _BYTE v50[96]; // [rsp+30h] [rbp-60h] BYREF

  v42 = (_QWORD *)a1[2];
  if ( (_QWORD *)a1[1] != v42 )
  {
    v6 = (_QWORD *)a1[1];
    v7 = 3;
    while ( 1 )
    {
      LOBYTE(v7) = (*(__int64 (__fastcall **)(_QWORD, unsigned __int8 *, unsigned __int8 *, __int64))(*(_QWORD *)*v6 + 64LL))(
                     *v6,
                     a2,
                     a3,
                     a4)
                 & v7;
      if ( !(_BYTE)v7 )
        return 0;
      if ( v42 == ++v6 )
        goto LABEL_8;
    }
  }
  v7 = 3;
LABEL_8:
  v9 = sub_CF5230((__int64)a1, (__int64)a2, a4);
  if ( !v9 )
    return 0;
  v10 = sub_CF5230((__int64)a1, (__int64)a3, a4);
  if ( !v10 )
    return 0;
  if ( (((unsigned __int8)(v9 >> 6) | (unsigned __int8)((v9 >> 4) | v9 | (v9 >> 2))) & 2) != 0 )
  {
    if ( (((v9 >> 6) | (v9 >> 4) | v9 | (v9 >> 2)) & 1) == 0 )
      v7 &= 2u;
  }
  else
  {
    if ( (((unsigned __int8)(v10 >> 6) | (unsigned __int8)((v10 >> 4) | v10 | (v10 >> 2))) & 2) == 0 )
      return 0;
    v7 &= 1u;
  }
  if ( (v10 & 0xFFFFFFFC) != 0 )
  {
    if ( (v9 & 0xFFFFFFFC) != 0 )
      return v7;
    v11 = *a2;
    v12 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    if ( v11 == 40 )
    {
      v47 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v24 = sub_B491D0((__int64)a2);
      v12 = v47;
      v13 = -32 - 32LL * v24;
    }
    else
    {
      v13 = -32;
      if ( v11 != 85 )
      {
        v13 = -96;
        if ( v11 != 34 )
          goto LABEL_71;
      }
    }
    if ( (a2[7] & 0x80u) != 0 )
    {
      v39 = v13;
      v48 = v12;
      v25 = sub_BD2BC0((__int64)a2);
      v12 = v48;
      v13 = v39;
      v27 = v25 + v26;
      if ( (a2[7] & 0x80u) == 0 )
      {
        if ( !(unsigned int)(v27 >> 4) )
          goto LABEL_50;
      }
      else
      {
        v28 = sub_BD2BC0((__int64)a2);
        v12 = v48;
        v13 = v39;
        if ( !(unsigned int)((v27 - v28) >> 4) )
          goto LABEL_50;
        if ( (a2[7] & 0x80u) != 0 )
        {
          v29 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
          if ( (a2[7] & 0x80u) == 0 )
            BUG();
          v30 = sub_BD2BC0((__int64)a2);
          v12 = v48;
          v13 = v39 - 32LL * (unsigned int)(*(_DWORD *)(v30 + v31 - 4) - v29);
          goto LABEL_50;
        }
      }
      BUG();
    }
LABEL_50:
    v32 = &a2[v13];
    if ( v12 != v32 )
    {
      v49 = 0;
      while ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v12 + 8LL) + 8LL) != 14 )
      {
LABEL_52:
        v12 += 32;
        if ( v12 == v32 )
          return v49;
      }
      v35 = v32;
      v36 = v12;
      v40 = (v12 - &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)]) >> 5;
      sub_D669C0(v50, a2, (unsigned int)v40, *a1);
      v41 = sub_CF51C0((__int64)a1, (__int64)a2, v40);
      v33 = sub_CF52B0(a1, a3, (__int64)v50, a4);
      v12 = v36;
      v32 = v35;
      if ( (v41 & 2) != 0 )
      {
        if ( !v33 )
          goto LABEL_57;
      }
      else if ( (v41 & 1) == 0 || (v33 & 2) == 0 )
      {
LABEL_57:
        if ( v49 == (_BYTE)v7 )
          return v7;
        goto LABEL_52;
      }
      v49 = v7 & (v49 | v41);
      goto LABEL_57;
    }
    return 0;
  }
  v14 = *a3;
  v15 = &a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
  if ( v14 == 40 )
  {
    v16 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a3);
  }
  else
  {
    v16 = -32;
    if ( v14 != 85 )
    {
      v16 = -96;
      if ( v14 != 34 )
LABEL_71:
        BUG();
    }
  }
  if ( (a3[7] & 0x80u) != 0 )
  {
    v43 = v16;
    v17 = sub_BD2BC0((__int64)a3);
    v16 = v43;
    if ( (a3[7] & 0x80u) == 0 )
    {
      if ( !(unsigned int)((v17 + v18) >> 4) )
        goto LABEL_33;
    }
    else
    {
      v37 = v43;
      v44 = v17 + v18;
      v19 = sub_BD2BC0((__int64)a3);
      v16 = v37;
      if ( !(unsigned int)((v44 - v19) >> 4) )
        goto LABEL_33;
      if ( (a3[7] & 0x80u) != 0 )
      {
        v45 = *(_DWORD *)(sub_BD2BC0((__int64)a3) + 8);
        if ( (a3[7] & 0x80u) == 0 )
          BUG();
        v20 = sub_BD2BC0((__int64)a3);
        v16 = v37 - 32LL * (unsigned int)(*(_DWORD *)(v20 + v21 - 4) - v45);
        goto LABEL_33;
      }
    }
    BUG();
  }
LABEL_33:
  v46 = &a3[v16];
  if ( v15 == &a3[v16] )
    return 0;
  v38 = 0;
  while ( 1 )
  {
    if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v15 + 8LL) + 8LL) == 14 )
    {
      v34 = (v15 - &a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)]) >> 5;
      sub_D669C0(v50, a3, (unsigned int)v34, *a1);
      v22 = sub_CF51C0((__int64)a1, (__int64)a3, v34);
      v23 = v22 & 2;
      if ( (v22 & 2) != 0 )
      {
        v23 = 3;
      }
      else if ( (v22 & 1) != 0 )
      {
        v23 = 2;
      }
      v38 = v7 & (v38 | v23 & sub_CF52B0(a1, a2, (__int64)v50, a4));
      if ( v38 == (_BYTE)v7 )
        break;
    }
    v15 += 32;
    if ( v15 == v46 )
      return v38;
  }
  return v7;
}
