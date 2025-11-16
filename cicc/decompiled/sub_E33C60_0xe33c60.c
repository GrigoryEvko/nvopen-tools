// Function: sub_E33C60
// Address: 0xe33c60
//
char __fastcall sub_E33C60(__int64 *a1, __int64 a2)
{
  _QWORD *v2; // r9
  __int64 v3; // r12
  __int64 v5; // rax
  int v6; // edx
  __int64 v7; // r13
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r13
  unsigned __int8 *v12; // r15
  const void *v13; // r9
  unsigned __int8 v14; // dl
  __int64 v15; // r12
  __int64 *v16; // rdx
  __int64 v17; // rdi
  __int64 *v18; // rdi
  __int64 v19; // r14
  unsigned __int8 v20; // al
  __int64 v21; // rax
  __int64 v22; // rcx
  int v23; // edx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r12
  __int64 v27; // rax
  int v28; // r12d
  __int64 v29; // rax
  __int64 v30; // rdx
  unsigned __int8 v31; // cl
  bool v32; // si
  int v33; // eax
  __int64 v34; // rdx
  __int64 v35; // r12
  __int64 v36; // r13
  __int64 v37; // rcx
  __int64 v38; // rcx
  __int64 v39; // r8
  unsigned int *v40; // rsi
  unsigned __int8 v41; // dl
  __int64 v42; // rdx
  __int64 v43; // rax
  __int64 v44; // r8
  unsigned int v45; // r12d
  unsigned int v46; // r14d
  __int64 v47; // r9
  __int16 v48; // di
  bool v49; // al
  int v50; // edx
  unsigned __int64 v51; // rdx
  __int64 v52; // rcx
  unsigned int v53; // eax
  __int64 v55; // [rsp+0h] [rbp-60h]
  unsigned int v56; // [rsp+8h] [rbp-58h]
  int v57; // [rsp+Ch] [rbp-54h]
  const void *v58; // [rsp+10h] [rbp-50h]
  const void *v59; // [rsp+20h] [rbp-40h]
  __int64 v60; // [rsp+20h] [rbp-40h]
  const void *v61; // [rsp+20h] [rbp-40h]
  const void *v62; // [rsp+28h] [rbp-38h]
  __int64 v63; // [rsp+28h] [rbp-38h]

  v2 = a1 + 3;
  v3 = a2;
  v5 = *(_QWORD *)(a2 + 24);
  v6 = *(unsigned __int8 *)v5;
  if ( (unsigned __int8)v6 > 0x1Cu
    && (v9 = (unsigned int)(v6 - 34), (unsigned __int8)v9 <= 0x33u)
    && (v10 = 0x8000000000041LL, _bittest64(&v10, v9)) )
  {
    *a1 = v5;
    a1[1] = (__int64)v2;
    a1[2] = 0;
  }
  else
  {
    a1[1] = (__int64)v2;
    a1[2] = 0;
    v7 = *(_QWORD *)(a2 + 24);
    *a1 = 0;
    if ( *(_BYTE *)v7 != 5 )
      return v5;
    v8 = *(_QWORD *)(v7 + 16);
    if ( v8 && !*(_QWORD *)(v8 + 8) && (v49 = sub_AC35E0(v7), v2 = a1 + 3, v49) )
    {
      v3 = *(_QWORD *)(v7 + 16);
      v5 = *(_QWORD *)(v3 + 24);
      v50 = *(unsigned __int8 *)v5;
      if ( (unsigned __int8)v50 <= 0x1Cu
        || (v51 = (unsigned int)(v50 - 34), (unsigned __int8)v51 > 0x33u)
        || (v52 = 0x8000000000041LL, !_bittest64(&v52, v51)) )
      {
LABEL_13:
        *a1 = 0;
        return v5;
      }
      *a1 = v5;
    }
    else
    {
      v5 = *a1;
      if ( !*a1 )
        return v5;
    }
  }
  if ( v3 == v5 - 32 )
    return v5;
  v11 = *(_QWORD *)(v5 - 32);
  if ( !v11 )
    goto LABEL_13;
  if ( *(_BYTE *)v11 )
    goto LABEL_13;
  if ( *(_QWORD *)(v5 + 80) != *(_QWORD *)(v11 + 24) )
    goto LABEL_13;
  v62 = v2;
  if ( (*(_BYTE *)(v11 + 7) & 0x20) == 0 )
    goto LABEL_13;
  v5 = sub_B91C10(v11, 26);
  if ( !v5 )
    goto LABEL_13;
  v12 = (unsigned __int8 *)*a1;
  v13 = v62;
  v14 = *(_BYTE *)(v5 - 16);
  v15 = (v3 - (*a1 - 32LL * (*(_DWORD *)(*a1 + 4) & 0x7FFFFFF))) >> 5;
  if ( (v14 & 2) != 0 )
  {
    v16 = *(__int64 **)(v5 - 32);
    v17 = *(unsigned int *)(v5 - 24);
  }
  else
  {
    v48 = *(_WORD *)(v5 - 16) >> 6;
    v5 -= 8LL * ((v14 >> 2) & 0xF);
    v17 = v48 & 0xF;
    v16 = (__int64 *)(v5 - 16);
  }
  v18 = &v16[v17];
  if ( v16 == v18 )
    goto LABEL_13;
  while ( 1 )
  {
    v19 = *v16;
    v20 = *(_BYTE *)(*v16 - 16);
    if ( (v20 & 2) != 0 )
      v21 = *(_QWORD *)(v19 - 32);
    else
      v21 = v19 + -16 - 8LL * ((v20 >> 2) & 0xF);
    v22 = *(_QWORD *)(*(_QWORD *)v21 + 136LL);
    v5 = *(_QWORD *)(v22 + 24);
    if ( *(_DWORD *)(v22 + 32) > 0x40u )
      v5 = *(_QWORD *)v5;
    if ( v5 == (unsigned int)v15 )
      break;
    if ( v18 == ++v16 )
      goto LABEL_13;
  }
  v23 = *v12;
  if ( v23 == 40 )
  {
    v53 = sub_B491D0(*a1);
    v13 = v62;
    v63 = 32LL * v53;
  }
  else
  {
    v63 = 0;
    if ( v23 != 85 )
    {
      v63 = 64;
      if ( v23 != 34 )
        BUG();
    }
  }
  if ( (v12[7] & 0x80u) != 0 )
  {
    v59 = v13;
    v24 = sub_BD2BC0((__int64)v12);
    v13 = v59;
    v26 = v24 + v25;
    if ( (v12[7] & 0x80u) == 0 )
    {
      if ( (unsigned int)(v26 >> 4) )
        goto LABEL_77;
    }
    else
    {
      v27 = sub_BD2BC0((__int64)v12);
      v13 = v59;
      if ( (unsigned int)((v26 - v27) >> 4) )
      {
        if ( (v12[7] & 0x80u) != 0 )
        {
          v28 = *(_DWORD *)(sub_BD2BC0((__int64)v12) + 8);
          if ( (v12[7] & 0x80u) == 0 )
            BUG();
          v29 = sub_BD2BC0((__int64)v12);
          v13 = v59;
          v55 = 32LL * (unsigned int)(*(_DWORD *)(v29 + v30 - 4) - v28);
          goto LABEL_36;
        }
LABEL_77:
        BUG();
      }
    }
  }
  v55 = 0;
LABEL_36:
  v31 = *(_BYTE *)(v19 - 16);
  v56 = *((_DWORD *)v12 + 1) & 0x7FFFFFF;
  v32 = (v31 & 2) != 0;
  if ( (v31 & 2) != 0 )
    v33 = *(_DWORD *)(v19 - 24);
  else
    v33 = (*(_WORD *)(v19 - 16) >> 6) & 0xF;
  if ( v33 != 1 )
  {
    v60 = v11;
    v34 = *((unsigned int *)a1 + 4);
    v35 = 0;
    v36 = 8LL * (unsigned int)(v33 - 2);
    while ( 1 )
    {
      v37 = v32 ? *(_QWORD *)(v19 - 32) : v19 + -16 - 8LL * ((v31 >> 2) & 0xF);
      v38 = *(_QWORD *)(*(_QWORD *)(v37 + v35) + 136LL);
      v39 = *(unsigned int *)(v38 + 32);
      v40 = *(unsigned int **)(v38 + 24);
      if ( (unsigned int)v39 > 0x40 )
      {
        v39 = *v40;
      }
      else if ( (_DWORD)v39 )
      {
        v39 = (__int64)((_QWORD)v40 << (64 - (unsigned __int8)v39)) >> (64 - (unsigned __int8)v39);
      }
      if ( v34 + 1 > (unsigned __int64)*((unsigned int *)a1 + 5) )
      {
        v57 = v39;
        v58 = v13;
        sub_C8D5F0((__int64)(a1 + 1), v13, v34 + 1, 4u, v39, (__int64)v13);
        v34 = *((unsigned int *)a1 + 4);
        LODWORD(v39) = v57;
        v13 = v58;
      }
      *(_DWORD *)(a1[1] + 4 * v34) = v39;
      v34 = (unsigned int)(*((_DWORD *)a1 + 4) + 1);
      *((_DWORD *)a1 + 4) = v34;
      if ( v36 == v35 )
        break;
      v31 = *(_BYTE *)(v19 - 16);
      v35 += 8;
      v32 = (v31 & 2) != 0;
    }
    v11 = v60;
  }
  LODWORD(v5) = *(_DWORD *)(*(_QWORD *)(v11 + 24) + 8LL) >> 8;
  if ( (_DWORD)v5 )
  {
    v41 = *(_BYTE *)(v19 - 16);
    if ( (v41 & 2) != 0 )
    {
      v42 = *(_QWORD *)(v19 - 32);
      v43 = (unsigned int)(*(_DWORD *)(v19 - 24) - 1);
    }
    else
    {
      v43 = ((*(_WORD *)(v19 - 16) >> 6) & 0xFu) - 1;
      v42 = v19 - 8LL * ((v41 >> 2) & 0xF) - 16;
    }
    v61 = v13;
    LOBYTE(v5) = sub_AC30F0(*(_QWORD *)(*(_QWORD *)(v42 + 8 * v43) + 136LL));
    if ( !(_BYTE)v5 )
    {
      v45 = *(_QWORD *)(v11 + 104);
      v5 = (32LL * v56 - 32 - v63 - v55) >> 5;
      v46 = v5;
      if ( v45 < (unsigned int)v5 )
      {
        v47 = (__int64)v61;
        v5 = *((unsigned int *)a1 + 4);
        do
        {
          if ( v5 + 1 > (unsigned __int64)*((unsigned int *)a1 + 5) )
          {
            sub_C8D5F0((__int64)(a1 + 1), v61, v5 + 1, 4u, v44, v47);
            v5 = *((unsigned int *)a1 + 4);
          }
          *(_DWORD *)(a1[1] + 4 * v5) = v45++;
          v5 = (unsigned int)(*((_DWORD *)a1 + 4) + 1);
          *((_DWORD *)a1 + 4) = v5;
        }
        while ( v45 < v46 );
      }
    }
  }
  return v5;
}
