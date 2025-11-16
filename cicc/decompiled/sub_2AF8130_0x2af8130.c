// Function: sub_2AF8130
// Address: 0x2af8130
//
__int64 __fastcall sub_2AF8130(__int64 a1, __int64 a2)
{
  char *v2; // rbx
  __int64 v3; // rax
  __int64 v4; // rax
  _QWORD *v6; // rsi
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdx
  char *v10; // r15
  __int64 v11; // r14
  __int64 *v12; // r13
  int v13; // ecx
  __int64 *v14; // rax
  unsigned int v15; // r14d
  __int64 v16; // r15
  _QWORD *v17; // rax
  unsigned int v18; // ecx
  __int64 v19; // r14
  unsigned int v20; // ecx
  __int64 v21; // r10
  _QWORD *v22; // rax
  __int64 *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // r13
  __int64 *v27; // rax
  __int64 v28; // rax
  __int64 *v29; // rdx
  __int64 v30; // rax
  __int64 v31; // r12
  __int64 v32; // rdx
  _QWORD *v33; // rcx
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rdi
  unsigned __int64 v37; // rcx
  __int16 v38; // ax
  unsigned __int64 v39; // rdx
  __int64 v41; // rax
  _QWORD *v42; // rcx
  __int64 *v43; // r8
  __int64 *v44; // rax
  __int64 v45; // rsi
  int v46; // edx
  char v47; // al
  __int64 v48; // rax
  unsigned int v49; // [rsp+4h] [rbp-FCh]
  _QWORD *v51; // [rsp+10h] [rbp-F0h]
  __int64 v52; // [rsp+18h] [rbp-E8h]
  __int64 v53; // [rsp+20h] [rbp-E0h]
  __int64 v54; // [rsp+28h] [rbp-D8h]
  __int64 v55; // [rsp+38h] [rbp-C8h]
  __int64 v56; // [rsp+48h] [rbp-B8h]
  _QWORD v57[4]; // [rsp+50h] [rbp-B0h] BYREF
  char v58; // [rsp+70h] [rbp-90h]
  char v59; // [rsp+71h] [rbp-8Fh]
  __int64 *v60; // [rsp+80h] [rbp-80h] BYREF
  __int64 v61; // [rsp+88h] [rbp-78h]
  _BYTE v62[112]; // [rsp+90h] [rbp-70h] BYREF

  v2 = sub_2AF7450((char *)a2);
  v3 = *((_DWORD *)v2 + 1) & 0x7FFFFFF;
  v52 = (unsigned int)(v3 - 1);
  v4 = *(_QWORD *)&v2[32 * (v52 - v3)];
  if ( *(_BYTE *)v4 != 17 )
    BUG();
  v6 = *(_QWORD **)(v4 + 24);
  if ( *(_DWORD *)(v4 + 32) > 0x40u )
    v6 = (_QWORD *)*v6;
  v54 = sub_AD64C0(*(_QWORD *)(v4 + 8), (__int64)v6 - 1, 0);
  v9 = *((_DWORD *)v2 + 1) & 0x7FFFFFF;
  v60 = (__int64 *)v62;
  v61 = 0x800000000LL;
  v10 = &v2[32 * (1 - v9)];
  v11 = (-32 * (1 - v9)) >> 5;
  if ( (unsigned __int64)(-32 * (1 - v9)) > 0x100 )
  {
    sub_C8D5F0((__int64)&v60, v62, (-32 * (1 - v9)) >> 5, 8u, v7, v8);
    v12 = v60;
    v13 = v61;
    v14 = &v60[(unsigned int)v61];
  }
  else
  {
    v12 = (__int64 *)v62;
    v13 = 0;
    v14 = (__int64 *)v62;
  }
  if ( v2 != v10 )
  {
    do
    {
      if ( v14 )
        *v14 = *(_QWORD *)v10;
      v10 += 32;
      ++v14;
    }
    while ( v2 != v10 );
    v12 = v60;
    v13 = v61;
  }
  v15 = v13 + v11;
  v59 = 1;
  LODWORD(v61) = v15;
  v16 = v15;
  v57[0] = "GapLoadGEP";
  v58 = 3;
  v55 = *(_QWORD *)&v2[-32 * (*((_DWORD *)v2 + 1) & 0x7FFFFFF)];
  v53 = sub_BB5290((__int64)v2);
  v17 = sub_BD2C40(88, v15 + 1);
  v18 = v15 + 1;
  v19 = (__int64)v17;
  if ( v17 )
  {
    v51 = v17;
    v20 = v18 & 0x7FFFFFF;
    v21 = *(_QWORD *)(v55 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v21 + 8) - 17 > 1 )
    {
      v43 = &v12[v16];
      if ( v12 != v43 )
      {
        v44 = v12;
        v45 = *(_QWORD *)(*v12 + 8);
        v46 = *(unsigned __int8 *)(v45 + 8);
        if ( v46 == 17 )
        {
LABEL_51:
          v47 = 0;
        }
        else
        {
          while ( v46 != 18 )
          {
            if ( v43 == ++v44 )
              goto LABEL_13;
            v45 = *(_QWORD *)(*v44 + 8);
            v46 = *(unsigned __int8 *)(v45 + 8);
            if ( v46 == 17 )
              goto LABEL_51;
          }
          v47 = 1;
        }
        BYTE4(v56) = v47;
        v49 = v20;
        LODWORD(v56) = *(_DWORD *)(v45 + 32);
        v48 = sub_BCE1B0((__int64 *)v21, v56);
        v20 = v49;
        v21 = v48;
      }
    }
LABEL_13:
    sub_B44260(v19, v21, 34, v20, 0, 0);
    *(_QWORD *)(v19 + 72) = v53;
    *(_QWORD *)(v19 + 80) = sub_B4DC50(v53, (__int64)v12, v16);
    sub_B4D9A0(v19, v55, v12, v16, (__int64)v57);
  }
  else
  {
    v51 = 0;
  }
  sub_B4DE00(v19, (v2[1] & 2) != 0);
  if ( (*(_BYTE *)(v19 + 7) & 0x40) != 0 )
    v22 = *(_QWORD **)(v19 - 8);
  else
    v22 = &v51[-4 * (*(_DWORD *)(v19 + 4) & 0x7FFFFFF)];
  v23 = &v22[4 * v52];
  if ( *v23 )
  {
    v24 = v23[1];
    *(_QWORD *)v23[2] = v24;
    if ( v24 )
      *(_QWORD *)(v24 + 16) = v23[2];
  }
  *v23 = v54;
  if ( v54 )
  {
    v25 = *(_QWORD *)(v54 + 16);
    v23[1] = v25;
    if ( v25 )
      *(_QWORD *)(v25 + 16) = v23 + 1;
    v23[2] = v54 + 16;
    *(_QWORD *)(v54 + 16) = v23;
  }
  v26 = a2 + 24;
  sub_B44220((_QWORD *)v19, a2 + 24, 0);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v27 = *(__int64 **)(a2 - 8);
    if ( *(_QWORD *)(*v27 + 8) != *(_QWORD *)(v19 + 8) )
    {
      v59 = 1;
      v57[0] = "GapLoadCast";
      v58 = 3;
      v28 = *v27;
LABEL_29:
      v19 = sub_B52190(v19, *(_QWORD *)(v28 + 8), (__int64)v57, 0, 0);
      sub_B44220((_QWORD *)v19, a2 + 24, 0);
      v30 = sub_B47F80((_BYTE *)a2);
      v31 = v30;
      v32 = v30 - 32;
      if ( !*(_QWORD *)(v30 - 32) || (v33 = *(_QWORD **)(v30 - 16), v34 = *(_QWORD *)(v30 - 24), (*v33 = v34) == 0) )
      {
LABEL_32:
        *(_QWORD *)(v31 - 32) = v19;
        if ( !v19 )
          goto LABEL_36;
        goto LABEL_33;
      }
LABEL_31:
      *(_QWORD *)(v34 + 16) = *(_QWORD *)(v31 - 16);
      goto LABEL_32;
    }
  }
  else
  {
    v29 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    v28 = *v29;
    if ( *(_QWORD *)(*v29 + 8) != *(_QWORD *)(v19 + 8) )
    {
      v59 = 1;
      v57[0] = "GapLoadCast";
      v58 = 3;
      goto LABEL_29;
    }
  }
  v41 = sub_B47F80((_BYTE *)a2);
  v31 = v41;
  v32 = v41 - 32;
  if ( *(_QWORD *)(v41 - 32) )
  {
    v42 = *(_QWORD **)(v41 - 16);
    v34 = *(_QWORD *)(v41 - 24);
    *v42 = v34;
    if ( v34 )
      goto LABEL_31;
  }
  *(_QWORD *)(v31 - 32) = v19;
LABEL_33:
  v35 = *(_QWORD *)(v19 + 16);
  *(_QWORD *)(v31 - 24) = v35;
  if ( v35 )
    *(_QWORD *)(v35 + 16) = v31 - 24;
  *(_QWORD *)(v31 - 16) = v19 + 16;
  *(_QWORD *)(v19 + 16) = v32;
LABEL_36:
  v36 = *(_QWORD *)(v31 + 8);
  if ( *(_BYTE *)(v36 + 8) == 14 )
  {
    *(_WORD *)(v31 + 2) = (2 * (unsigned __int8)sub_AE5020(*(_QWORD *)(a1 + 48), *(_QWORD *)(v31 + 8)))
                        | *(_WORD *)(v31 + 2) & 0xFF81;
  }
  else
  {
    v37 = (unsigned int)sub_BCB060(v36) >> 3;
    v38 = 510;
    if ( (_DWORD)v37 )
    {
      _BitScanReverse64(&v39, v37);
      v38 = (2 * (63 - (v39 ^ 0x3F))) & 0x1FE;
    }
    *(_WORD *)(v31 + 2) = *(_WORD *)(v31 + 2) & 0xFF81 | v38;
  }
  sub_B44220((_QWORD *)v31, v26, 0);
  if ( v60 != (__int64 *)v62 )
    _libc_free((unsigned __int64)v60);
  return v31;
}
