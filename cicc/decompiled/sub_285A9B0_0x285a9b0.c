// Function: sub_285A9B0
// Address: 0x285a9b0
//
char __fastcall sub_285A9B0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  __int64 v4; // r14
  __int64 v5; // rbx
  __int64 v6; // r15
  _BYTE **v7; // rcx
  _BYTE *v8; // r11
  __int64 v9; // r12
  bool v10; // zf
  __int64 v11; // rax
  bool v12; // si
  unsigned int v13; // edi
  __int64 v14; // rax
  double v15; // xmm0_8
  unsigned __int8 **v16; // r11
  _BYTE *v17; // rsi
  __int64 v18; // rdx
  unsigned int v19; // r8d
  __int64 v20; // rdi
  __int64 v21; // r13
  __int64 v22; // r15
  __int64 v23; // rt0
  __int64 v24; // rbx
  __int64 v25; // rax
  unsigned __int8 **v26; // r11
  __int64 v27; // r14
  __int64 v28; // rax
  double v29; // xmm0_8
  unsigned __int8 *v30; // rax
  __int64 v31; // rax
  unsigned __int8 **v32; // r11
  __int64 v33; // r13
  __int64 v34; // rt1
  unsigned __int8 **v36; // [rsp+10h] [rbp-90h]
  unsigned __int8 **v37; // [rsp+10h] [rbp-90h]
  unsigned __int8 **v38; // [rsp+10h] [rbp-90h]
  unsigned __int8 **v39; // [rsp+10h] [rbp-90h]
  unsigned __int8 **v40; // [rsp+10h] [rbp-90h]
  _BOOL8 v41; // [rsp+18h] [rbp-88h]
  _BYTE *v42; // [rsp+20h] [rbp-80h]
  _BOOL8 v43; // [rsp+20h] [rbp-80h]
  __int64 v44; // [rsp+28h] [rbp-78h]
  _BYTE *v45; // [rsp+28h] [rbp-78h]
  __int64 v46; // [rsp+28h] [rbp-78h]
  _BYTE *v47; // [rsp+30h] [rbp-70h]
  int v48; // [rsp+30h] [rbp-70h]
  __int64 v49; // [rsp+30h] [rbp-70h]
  _BYTE *v50; // [rsp+30h] [rbp-70h]
  __int64 v51; // [rsp+30h] [rbp-70h]
  char v52; // [rsp+38h] [rbp-68h]
  _BOOL4 v53; // [rsp+38h] [rbp-68h]
  unsigned __int8 **v54; // [rsp+38h] [rbp-68h]
  unsigned __int8 **v55; // [rsp+38h] [rbp-68h]
  unsigned __int8 **v56; // [rsp+38h] [rbp-68h]
  int v57; // [rsp+38h] [rbp-68h]
  unsigned __int8 *v58[4]; // [rsp+40h] [rbp-60h] BYREF
  char v59; // [rsp+60h] [rbp-40h]
  char v60; // [rsp+61h] [rbp-3Fh]

  v1 = sub_DCF3A0(*(__int64 **)(a1 + 8), *(char **)(a1 + 56), 0);
  LOBYTE(v2) = sub_D96A50(v1);
  if ( (_BYTE)v2 )
    return v2;
  v4 = *(_QWORD *)(*(_QWORD *)a1 + 208LL);
  v5 = *(_QWORD *)a1 + 200LL;
  if ( v4 == v5 )
    return v2;
  while ( 1 )
  {
    v2 = v4;
    v4 = *(_QWORD *)(v4 + 8);
    v9 = *(_QWORD *)(v2 - 8);
    LOBYTE(v2) = *(_BYTE *)v9;
    if ( *(_BYTE *)v9 == 72 )
    {
      v52 = 0;
      v6 = *(_QWORD *)(v9 + 8);
    }
    else
    {
      if ( (_BYTE)v2 != 73 )
        goto LABEL_9;
      v52 = 1;
      v6 = *(_QWORD *)(v9 + 8);
    }
    if ( !v6 )
      goto LABEL_9;
    LOBYTE(v2) = sub_DFA8F0(*(_QWORD *)(a1 + 48));
    if ( !(_BYTE)v2 )
      goto LABEL_9;
    if ( (*(_BYTE *)(v9 + 7) & 0x40) != 0 )
    {
      v7 = *(_BYTE ***)(v9 - 8);
      v8 = *v7;
      if ( **v7 != 84 )
        goto LABEL_9;
    }
    else
    {
      v2 = 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF);
      v8 = *(_BYTE **)(v9 - v2);
      if ( *v8 != 84 )
        goto LABEL_9;
    }
    LODWORD(v2) = *((_DWORD *)v8 + 1) & 0x7FFFFFF;
    if ( (_DWORD)v2 != 2 )
      goto LABEL_9;
    v47 = v8;
    v2 = (__int64)sub_DD8400(*(_QWORD *)(a1 + 8), (__int64)v8);
    if ( *(_WORD *)(v2 + 24) != 8 )
      goto LABEL_9;
    LOWORD(v2) = *(_WORD *)(v2 + 28);
    if ( v52 )
    {
      if ( (v2 & 4) == 0 )
        goto LABEL_9;
    }
    else if ( (v2 & 2) == 0 )
    {
      goto LABEL_9;
    }
    v42 = v47;
    v44 = *((_QWORD *)v47 + 1);
    LODWORD(v2) = sub_BCB090(v6);
    v48 = v2;
    if ( (_DWORD)v2 == -1 )
      goto LABEL_9;
    LODWORD(v2) = sub_D97050(*(_QWORD *)(a1 + 8), v44);
    if ( v48 < (int)v2 )
      goto LABEL_9;
    v45 = v42;
    v49 = *(_QWORD *)(*((_QWORD *)v42 - 1) + 32LL * *((unsigned int *)v42 + 18));
    v10 = sub_D4B130(*(_QWORD *)(a1 + 56)) == v49;
    v11 = 0;
    if ( !v10 )
      v11 = 32;
    v12 = v10;
    v43 = !v10;
    v2 = *(_QWORD *)(*((_QWORD *)v45 - 1) + v11);
    if ( *(_BYTE *)v2 != 17 )
      goto LABEL_9;
    v13 = *(_DWORD *)(v2 + 32);
    v14 = *(_QWORD *)(v2 + 24);
    if ( v52 )
    {
      if ( v13 > 0x40 )
      {
        v15 = (double)(int)*(_QWORD *)v14;
      }
      else
      {
        v15 = 0.0;
        if ( v13 )
          v15 = (double)(int)(v14 << (64 - (unsigned __int8)v13) >> (64 - (unsigned __int8)v13));
      }
    }
    else
    {
      if ( v13 > 0x40 )
        v14 = *(_QWORD *)v14;
      v15 = v14 < 0
          ? (double)(int)(v14 & 1 | ((unsigned __int64)v14 >> 1))
          + (double)(int)(v14 & 1 | ((unsigned __int64)v14 >> 1))
          : (double)(int)v14;
    }
    v50 = v45;
    v53 = v10;
    v2 = (__int64)sub_AD8DD0(v6, v15);
    v16 = (unsigned __int8 **)v45;
    v46 = v2;
    v41 = v12;
    v17 = *(_BYTE **)&(*(v16 - 1))[32 * v12];
    LOBYTE(v2) = *v17 - 42;
    if ( (unsigned __int8)v2 > 0x11u || (v2 & 0xFD) != 0 )
      goto LABEL_9;
    v18 = *((_QWORD *)v17 - 8);
    v2 = *((_QWORD *)v17 - 4);
    if ( v18 && v50 == (_BYTE *)v18 )
    {
      v18 = *((_QWORD *)v17 - 4);
      if ( *(_BYTE *)v2 != 17 )
        goto LABEL_9;
    }
    else if ( !v2 || v50 != (_BYTE *)v2 || *(_BYTE *)v18 != 17 )
    {
      goto LABEL_9;
    }
    v19 = *(_DWORD *)(v18 + 32);
    v20 = *(_QWORD *)(v18 + 24);
    v2 = 1LL << ((unsigned __int8)v19 - 1);
    if ( v19 > 0x40 )
      break;
    if ( (v2 & v20) == 0 && v20 )
    {
      v54 = *(unsigned __int8 ***)(*((_QWORD *)v50 - 1) + 32LL * v53);
      v23 = v6;
      v22 = a1;
      v21 = v23;
      v51 = v18;
      goto LABEL_38;
    }
LABEL_9:
    if ( v4 == v5 )
      return v2;
  }
  v2 &= *(_QWORD *)(v20 + 8LL * ((v19 - 1) >> 6));
  if ( v2 )
    goto LABEL_9;
  v57 = *(_DWORD *)(v18 + 32);
  v40 = (unsigned __int8 **)v50;
  v51 = v18;
  LODWORD(v2) = sub_C444A0(v18 + 24);
  if ( v57 == (_DWORD)v2 )
    goto LABEL_9;
  v16 = v40;
  v34 = v6;
  v22 = a1;
  v21 = v34;
  v54 = (unsigned __int8 **)v17;
LABEL_38:
  v24 = (__int64)(v16 + 3);
  v60 = 1;
  v36 = v16;
  v58[0] = "IV.S.";
  v59 = 3;
  v25 = sub_BD2DA0(80);
  v26 = v36;
  v27 = v25;
  if ( v25 )
  {
    sub_B44260(v25, v21, 55, 0x8000000u, v24, 0);
    *(_DWORD *)(v27 + 72) = 2;
    sub_BD6B50((unsigned __int8 *)v27, (const char **)v58);
    sub_BD2A10(v27, *(_DWORD *)(v27 + 72), 1);
    v26 = v36;
  }
  v58[0] = v26[6];
  if ( v58[0] )
  {
    v37 = v26;
    sub_2850C40((__int64 *)v58);
    v26 = v37;
  }
  if ( (unsigned __int8 **)(v27 + 48) != v58 )
  {
    v38 = v26;
    sub_2850F80((__int64 *)(v27 + 48), v58);
    v26 = v38;
  }
  v39 = v26;
  sub_9C6650(v58);
  if ( *(_DWORD *)(v51 + 32) <= 0x40u )
    v28 = *(_QWORD *)(v51 + 24);
  else
    v28 = **(_QWORD **)(v51 + 24);
  if ( v28 < 0 )
    v29 = (double)(int)(v28 & 1 | ((unsigned __int64)v28 >> 1)) + (double)(int)(v28 & 1 | ((unsigned __int64)v28 >> 1));
  else
    v29 = (double)(int)v28;
  v30 = sub_AD8DD0(v21, v29);
  v60 = 1;
  v59 = 3;
  v58[0] = "IV.S.next.";
  v31 = sub_B504D0(2 * (unsigned int)(*(_BYTE *)v54 != 42) + 14, v27, (__int64)v30, (__int64)v58, (__int64)(v54 + 3), 0);
  v32 = v39;
  v33 = v31;
  v58[0] = v54[6];
  if ( v58[0] )
  {
    sub_2850C40((__int64 *)v58);
    v32 = v39;
  }
  if ( (unsigned __int8 **)(v33 + 48) != v58 )
  {
    v55 = v32;
    sub_2850F80((__int64 *)(v33 + 48), v58);
    v32 = v55;
  }
  v56 = v32;
  sub_9C6650(v58);
  sub_F0A850(v27, v46, *(_QWORD *)&(*(v56 - 1))[32 * *((unsigned int *)v56 + 18) + 8 * v43]);
  sub_F0A850(v27, v33, *(_QWORD *)&(*(v56 - 1))[32 * *((unsigned int *)v56 + 18) + 8 * v41]);
  sub_BD84D0(v9, v27);
  LOBYTE(v2) = sub_B43D60((_QWORD *)v9);
  *(_BYTE *)(v22 + 952) = 1;
  return v2;
}
