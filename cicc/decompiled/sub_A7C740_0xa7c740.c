// Function: sub_A7C740
// Address: 0xa7c740
//
__int64 __fastcall sub_A7C740(__int64 a1, __int64 a2, unsigned __int8 *a3)
{
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // r13
  __int64 v7; // rdx
  __int64 v8; // rax
  _BYTE *v9; // r14
  __int64 v10; // rax
  _BYTE *v11; // r15
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // r9
  __int64 result; // rax
  _BYTE *v16; // rdi
  __int64 v17; // rax
  __int64 v18; // r14
  __int64 v19; // r15
  __int64 v20; // rax
  _BYTE *v21; // r13
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // r14
  __int64 v25; // rdx
  __int64 v26; // rax
  _BYTE *v27; // r13
  __int64 v28; // rax
  _BYTE *v29; // rcx
  __int64 v30; // rax
  _BYTE *v31; // r8
  __int64 v32; // rax
  __int64 v33; // r9
  __int64 v34; // rax
  _BYTE *v35; // r15
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // r13
  int v41; // r13d
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // rdx
  _BYTE *v46; // rdi
  __int64 v47; // rcx
  __int64 v48; // rax
  __int64 v49; // rsi
  __int64 v50; // rcx
  __int64 v51; // rax
  _BYTE *v52; // r13
  __int64 v53; // rax
  __int64 v54; // rsi
  __int64 v55; // rax
  int v56; // [rsp+0h] [rbp-60h]
  int v57; // [rsp+8h] [rbp-58h]
  int v58; // [rsp+10h] [rbp-50h]
  __int64 v59; // [rsp+18h] [rbp-48h]
  __int64 v60; // [rsp+18h] [rbp-48h]
  __int64 v61; // [rsp+18h] [rbp-48h]
  _QWORD v62[7]; // [rsp+28h] [rbp-38h] BYREF

  if ( a2 != 5 )
  {
    if ( a2 == 6 )
    {
      if ( *(_DWORD *)a1 == 1769173857 && *(_WORD *)(a1 + 4) == 28263 )
      {
        LODWORD(v24) = 0;
        v25 = *((_DWORD *)a3 + 1) & 0x7FFFFFF;
        v26 = *(_QWORD *)&a3[-32 * v25];
        if ( *(_BYTE *)v26 == 24 )
          v24 = *(_QWORD *)(v26 + 24);
        LODWORD(v27) = 0;
        v28 = *(_QWORD *)&a3[32 * (1 - v25)];
        if ( *(_BYTE *)v28 == 24 )
        {
          v27 = *(_BYTE **)(v28 + 24);
          if ( *v27 != 26 )
            LODWORD(v27) = 0;
        }
        LODWORD(v29) = 0;
        v30 = *(_QWORD *)&a3[32 * (2 - v25)];
        if ( *(_BYTE *)v30 == 24 )
        {
          v29 = *(_BYTE **)(v30 + 24);
          if ( *v29 != 7 )
            LODWORD(v29) = 0;
        }
        LODWORD(v31) = 0;
        v32 = *(_QWORD *)&a3[32 * (3 - v25)];
        if ( *(_BYTE *)v32 == 24 )
        {
          v31 = *(_BYTE **)(v32 + 24);
          if ( *v31 != 30 )
            LODWORD(v31) = 0;
        }
        LODWORD(v33) = 0;
        v34 = *(_QWORD *)&a3[32 * (4 - v25)];
        if ( *(_BYTE *)v34 == 24 )
          v33 = *(_QWORD *)(v34 + 24);
        v35 = 0;
        v36 = *(_QWORD *)&a3[32 * (5 - v25)];
        if ( *(_BYTE *)v36 == 24 )
        {
          v35 = *(_BYTE **)(v36 + 24);
          if ( *v35 != 7 )
            v35 = 0;
        }
        v56 = (int)v29;
        v57 = (int)v31;
        v58 = v33;
        v61 = sub_B10CD0(a3 + 48);
        v37 = sub_22077B0(96);
        v13 = v37;
        if ( v37 )
          sub_B12230(v37, v24, (_DWORD)v27, v56, v57, v58, (__int64)v35, v61);
        return sub_AA8770(*((_QWORD *)a3 + 5), v13, a3 + 24, 0);
      }
    }
    else if ( a2 == 7 )
    {
      if ( *(_DWORD *)a1 == 1818453348 && *(_WORD *)(a1 + 4) == 29281 && *(_BYTE *)(a1 + 6) == 101 )
      {
        v6 = 0;
        v7 = *((_DWORD *)a3 + 1) & 0x7FFFFFF;
        v8 = *(_QWORD *)&a3[-32 * v7];
        if ( *(_BYTE *)v8 == 24 )
          v6 = *(_QWORD *)(v8 + 24);
        v9 = 0;
        v10 = *(_QWORD *)&a3[32 * (1 - v7)];
        if ( *(_BYTE *)v10 == 24 )
        {
          v9 = *(_BYTE **)(v10 + 24);
          if ( *v9 != 26 )
            v9 = 0;
        }
        v11 = 0;
        v12 = *(_QWORD *)&a3[32 * (2 - v7)];
        if ( *(_BYTE *)v12 == 24 )
        {
          v11 = *(_BYTE **)(v12 + 24);
          if ( *v11 != 7 )
            v11 = 0;
        }
        v59 = sub_B10CD0(a3 + 48);
        v13 = sub_22077B0(96);
        if ( !v13 )
          return sub_AA8770(*((_QWORD *)a3 + 5), v13, a3 + 24, 0);
        v14 = 0;
LABEL_25:
        sub_B12150(v13, v6, v9, v11, v59, v14);
        return sub_AA8770(*((_QWORD *)a3 + 5), v13, a3 + 24, 0);
      }
    }
    else if ( a2 == 4 && *(_DWORD *)a1 == 1919181921 )
    {
      v16 = 0;
      v17 = *(_QWORD *)&a3[32 * (2LL - (*((_DWORD *)a3 + 1) & 0x7FFFFFF))];
      if ( *(_BYTE *)v17 == 24 )
      {
        v16 = *(_BYTE **)(v17 + 24);
        if ( *v16 != 7 )
          v16 = 0;
      }
      v62[0] = 6;
      v18 = 0;
      v19 = sub_B0DED0(v16, v62, 1);
      v20 = *(_QWORD *)&a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
      if ( *(_BYTE *)v20 == 24 )
        v18 = *(_QWORD *)(v20 + 24);
      v21 = 0;
      v22 = *(_QWORD *)&a3[32 * (1LL - (*((_DWORD *)a3 + 1) & 0x7FFFFFF))];
      if ( *(_BYTE *)v22 == 24 )
      {
        v21 = *(_BYTE **)(v22 + 24);
        if ( *v21 != 26 )
          v21 = 0;
      }
      v60 = sub_B10CD0(a3 + 48);
      v23 = sub_22077B0(96);
      v13 = v23;
      if ( v23 )
        sub_B12150(v23, v18, v21, v19, v60, 1);
      return sub_AA8770(*((_QWORD *)a3 + 5), v13, a3 + 24, 0);
    }
LABEL_27:
    v13 = 0;
    return sub_AA8770(*((_QWORD *)a3 + 5), v13, a3 + 24, 0);
  }
  if ( *(_DWORD *)a1 == 1700946284 && *(_BYTE *)(a1 + 4) == 108 )
  {
    v52 = 0;
    v53 = *(_QWORD *)&a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
    if ( *(_BYTE *)v53 == 24 )
    {
      v52 = *(_BYTE **)(v53 + 24);
      if ( *v52 != 27 )
        v52 = 0;
    }
    v54 = *((_QWORD *)a3 + 6);
    v62[0] = v54;
    if ( v54 )
      sub_B96E90(v62, v54, 1);
    v55 = sub_22077B0(48);
    v13 = v55;
    if ( v55 )
      sub_B12570(v55, v52, v62);
    if ( v62[0] )
      sub_B91220(v62);
    return sub_AA8770(*((_QWORD *)a3 + 5), v13, a3 + 24, 0);
  }
  if ( *(_DWORD *)a1 != 1970037110 || *(_BYTE *)(a1 + 4) != 101 )
    goto LABEL_27;
  v4 = *a3;
  if ( v4 == 40 )
  {
    v5 = 32LL * (unsigned int)sub_B491D0(a3);
  }
  else
  {
    v5 = 0;
    if ( v4 != 85 )
    {
      v5 = 64;
      if ( v4 != 34 )
LABEL_96:
        BUG();
    }
  }
  if ( (a3[7] & 0x80u) == 0 )
    goto LABEL_93;
  v38 = sub_BD2BC0(a3);
  v40 = v38 + v39;
  if ( (a3[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v40 >> 4) )
      goto LABEL_96;
    goto LABEL_93;
  }
  if ( !(unsigned int)((v40 - sub_BD2BC0(a3)) >> 4) )
  {
LABEL_93:
    v44 = 0;
    goto LABEL_68;
  }
  if ( (a3[7] & 0x80u) == 0 )
    goto LABEL_96;
  v41 = *(_DWORD *)(sub_BD2BC0(a3) + 8);
  if ( (a3[7] & 0x80u) == 0 )
    BUG();
  v42 = sub_BD2BC0(a3);
  v44 = 32LL * (unsigned int)(*(_DWORD *)(v42 + v43 - 4) - v41);
LABEL_68:
  v45 = *((_DWORD *)a3 + 1) & 0x7FFFFFF;
  if ( (unsigned int)((32 * v45 - 32 - v5 - v44) >> 5) != 4 )
  {
    v47 = 1;
    v48 = 2;
    goto LABEL_73;
  }
  result = 32 * (1 - v45);
  v46 = *(_BYTE **)&a3[result];
  if ( v46 )
  {
    if ( *v46 <= 0x15u )
    {
      result = sub_AD7890(v46);
      if ( (_BYTE)result )
      {
        v47 = 2;
        v48 = 3;
        v45 = *((_DWORD *)a3 + 1) & 0x7FFFFFF;
LABEL_73:
        v6 = 0;
        v49 = *(_QWORD *)&a3[-32 * v45];
        if ( *(_BYTE *)v49 == 24 )
          v6 = *(_QWORD *)(v49 + 24);
        v9 = 0;
        v50 = *(_QWORD *)&a3[32 * (v47 - v45)];
        if ( *(_BYTE *)v50 == 24 )
        {
          v9 = *(_BYTE **)(v50 + 24);
          if ( *v9 != 26 )
            v9 = 0;
        }
        v11 = 0;
        v51 = *(_QWORD *)&a3[32 * (v48 - v45)];
        if ( *(_BYTE *)v51 == 24 )
        {
          v11 = *(_BYTE **)(v51 + 24);
          if ( *v11 != 7 )
            v11 = 0;
        }
        v59 = sub_B10CD0(a3 + 48);
        v13 = sub_22077B0(96);
        if ( !v13 )
          return sub_AA8770(*((_QWORD *)a3 + 5), v13, a3 + 24, 0);
        v14 = 1;
        goto LABEL_25;
      }
    }
  }
  return result;
}
