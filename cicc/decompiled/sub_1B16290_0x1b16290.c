// Function: sub_1B16290
// Address: 0x1b16290
//
_QWORD *__fastcall sub_1B16290(__int64 a1, int a2, _BYTE *a3, __int64 a4)
{
  int v7; // r15d
  int v8; // eax
  __int64 v9; // rax
  __int64 v10; // r14
  _QWORD *v11; // r15
  _QWORD *v13; // rax
  _QWORD **v14; // rax
  __int64 *v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rsi
  __int64 v22; // rdx
  unsigned __int8 *v23; // rsi
  _QWORD *v24; // rax
  __int64 v25; // rdx
  unsigned __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  unsigned __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rdx
  unsigned __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdi
  unsigned __int64 *v35; // r12
  __int64 v36; // rax
  unsigned __int64 v37; // rcx
  __int64 v38; // rsi
  __int64 v39; // rsi
  unsigned __int8 *v40; // rsi
  _QWORD *v41; // rax
  _QWORD **v42; // rax
  __int64 *v43; // rax
  __int64 v44; // rsi
  __int64 v45; // rdx
  int v46; // r15d
  __int64 v47; // rdi
  __int64 *v48; // r15
  __int64 v49; // rax
  __int64 v50; // rsi
  _QWORD *v51; // [rsp+0h] [rbp-A0h]
  __int64 *v52; // [rsp+0h] [rbp-A0h]
  _QWORD *v53; // [rsp+0h] [rbp-A0h]
  _QWORD *v54; // [rsp+0h] [rbp-A0h]
  __int64 v55; // [rsp+8h] [rbp-98h]
  __int64 v56; // [rsp+8h] [rbp-98h]
  int v57; // [rsp+14h] [rbp-8Ch]
  __int64 v58; // [rsp+18h] [rbp-88h]
  unsigned __int8 *v59; // [rsp+28h] [rbp-78h] BYREF
  __int64 v60[2]; // [rsp+30h] [rbp-70h] BYREF
  char v61; // [rsp+40h] [rbp-60h]
  char v62; // [rsp+41h] [rbp-5Fh]
  __int64 v63[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v64; // [rsp+60h] [rbp-40h]

  v7 = dword_42C7340[a2 - 1];
  v8 = *(_DWORD *)(a1 + 40);
  v62 = 1;
  *(_DWORD *)(a1 + 40) = -1;
  v57 = v8;
  v9 = *(_QWORD *)(a1 + 32);
  v61 = 3;
  v58 = v9;
  v60[0] = (__int64)"rdx.minmax.cmp";
  if ( (unsigned int)(a2 - 5) <= 1 )
  {
    if ( a3[16] <= 0x10u && *(_BYTE *)(a4 + 16) <= 0x10u )
      goto LABEL_4;
    v64 = 257;
    v41 = sub_1648A60(56, 2u);
    v10 = (__int64)v41;
    if ( v41 )
    {
      v55 = (__int64)v41;
      v42 = *(_QWORD ***)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
      {
        v54 = v42[4];
        v43 = (__int64 *)sub_1643320(*v42);
        v44 = (__int64)sub_16463B0(v43, (unsigned int)v54);
      }
      else
      {
        v44 = sub_1643320(*v42);
      }
      sub_15FEC10(v10, v44, 52, v7, (__int64)a3, a4, (__int64)v63, 0);
    }
    else
    {
      v55 = 0;
    }
    v45 = *(_QWORD *)(a1 + 32);
    v46 = *(_DWORD *)(a1 + 40);
    if ( v45 )
      sub_1625C10(v10, 3, v45);
    sub_15F2440(v10, v46);
    v47 = *(_QWORD *)(a1 + 8);
    if ( v47 )
    {
      v48 = *(__int64 **)(a1 + 16);
      sub_157E9D0(v47 + 40, v10);
      v49 = *(_QWORD *)(v10 + 24);
      v50 = *v48;
      *(_QWORD *)(v10 + 32) = v48;
      v50 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v10 + 24) = v50 | v49 & 7;
      *(_QWORD *)(v50 + 8) = v10 + 24;
      *v48 = *v48 & 7 | (v10 + 24);
    }
  }
  else
  {
    if ( a3[16] <= 0x10u && *(_BYTE *)(a4 + 16) <= 0x10u )
    {
LABEL_4:
      v10 = sub_15A37B0(v7, a3, (_QWORD *)a4, 0);
      goto LABEL_5;
    }
    v64 = 257;
    v13 = sub_1648A60(56, 2u);
    v10 = (__int64)v13;
    if ( v13 )
    {
      v55 = (__int64)v13;
      v14 = *(_QWORD ***)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
      {
        v51 = v14[4];
        v15 = (__int64 *)sub_1643320(*v14);
        v16 = (__int64)sub_16463B0(v15, (unsigned int)v51);
      }
      else
      {
        v16 = sub_1643320(*v14);
      }
      sub_15FEC10(v10, v16, 51, v7, (__int64)a3, a4, (__int64)v63, 0);
    }
    else
    {
      v55 = 0;
    }
    v17 = *(_QWORD *)(a1 + 8);
    if ( v17 )
    {
      v52 = *(__int64 **)(a1 + 16);
      sub_157E9D0(v17 + 40, v10);
      v18 = *v52;
      v19 = *(_QWORD *)(v10 + 24) & 7LL;
      *(_QWORD *)(v10 + 32) = v52;
      v18 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v10 + 24) = v18 | v19;
      *(_QWORD *)(v18 + 8) = v10 + 24;
      *v52 = *v52 & 7 | (v10 + 24);
    }
  }
  sub_164B780(v55, v60);
  v20 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    v59 = *(unsigned __int8 **)a1;
    sub_1623A60((__int64)&v59, v20, 2);
    v21 = *(_QWORD *)(v10 + 48);
    v22 = v10 + 48;
    if ( v21 )
    {
      sub_161E7C0(v10 + 48, v21);
      v22 = v10 + 48;
    }
    v23 = v59;
    *(_QWORD *)(v10 + 48) = v59;
    if ( v23 )
      sub_1623210((__int64)&v59, v23, v22);
  }
LABEL_5:
  v62 = 1;
  v60[0] = (__int64)"rdx.minmax.select";
  v61 = 3;
  if ( *(_BYTE *)(v10 + 16) > 0x10u || a3[16] > 0x10u || *(_BYTE *)(a4 + 16) > 0x10u )
  {
    v64 = 257;
    v24 = sub_1648A60(56, 3u);
    v11 = v24;
    if ( v24 )
    {
      v53 = v24 - 9;
      v56 = (__int64)v24;
      sub_15F1EA0((__int64)v24, *(_QWORD *)a3, 55, (__int64)(v24 - 9), 3, 0);
      if ( *(v11 - 9) )
      {
        v25 = *(v11 - 8);
        v26 = *(v11 - 7) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v26 = v25;
        if ( v25 )
          *(_QWORD *)(v25 + 16) = *(_QWORD *)(v25 + 16) & 3LL | v26;
      }
      *(v11 - 9) = v10;
      v27 = *(_QWORD *)(v10 + 8);
      *(v11 - 8) = v27;
      if ( v27 )
        *(_QWORD *)(v27 + 16) = (unsigned __int64)(v11 - 8) | *(_QWORD *)(v27 + 16) & 3LL;
      *(v11 - 7) = (v10 + 8) | *(v11 - 7) & 3LL;
      *(_QWORD *)(v10 + 8) = v53;
      if ( *(v11 - 6) )
      {
        v28 = *(v11 - 5);
        v29 = *(v11 - 4) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v29 = v28;
        if ( v28 )
          *(_QWORD *)(v28 + 16) = *(_QWORD *)(v28 + 16) & 3LL | v29;
      }
      *(v11 - 6) = a3;
      v30 = *((_QWORD *)a3 + 1);
      *(v11 - 5) = v30;
      if ( v30 )
        *(_QWORD *)(v30 + 16) = (unsigned __int64)(v11 - 5) | *(_QWORD *)(v30 + 16) & 3LL;
      *(v11 - 4) = (unsigned __int64)(a3 + 8) | *(v11 - 4) & 3LL;
      *((_QWORD *)a3 + 1) = v11 - 6;
      if ( *(v11 - 3) )
      {
        v31 = *(v11 - 2);
        v32 = *(v11 - 1) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v32 = v31;
        if ( v31 )
          *(_QWORD *)(v31 + 16) = *(_QWORD *)(v31 + 16) & 3LL | v32;
      }
      *(v11 - 3) = a4;
      if ( a4 )
      {
        v33 = *(_QWORD *)(a4 + 8);
        *(v11 - 2) = v33;
        if ( v33 )
          *(_QWORD *)(v33 + 16) = (unsigned __int64)(v11 - 2) | *(_QWORD *)(v33 + 16) & 3LL;
        *(v11 - 1) = (a4 + 8) | *(v11 - 1) & 3LL;
        *(_QWORD *)(a4 + 8) = v11 - 3;
      }
      sub_164B780((__int64)v11, v63);
    }
    else
    {
      v56 = 0;
    }
    v34 = *(_QWORD *)(a1 + 8);
    if ( v34 )
    {
      v35 = *(unsigned __int64 **)(a1 + 16);
      sub_157E9D0(v34 + 40, (__int64)v11);
      v36 = v11[3];
      v37 = *v35;
      v11[4] = v35;
      v37 &= 0xFFFFFFFFFFFFFFF8LL;
      v11[3] = v37 | v36 & 7;
      *(_QWORD *)(v37 + 8) = v11 + 3;
      *v35 = *v35 & 7 | (unsigned __int64)(v11 + 3);
    }
    sub_164B780(v56, v60);
    v38 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
    {
      v63[0] = *(_QWORD *)a1;
      sub_1623A60((__int64)v63, v38, 2);
      v39 = v11[6];
      if ( v39 )
        sub_161E7C0((__int64)(v11 + 6), v39);
      v40 = (unsigned __int8 *)v63[0];
      v11[6] = v63[0];
      if ( v40 )
        sub_1623210((__int64)v63, v40, (__int64)(v11 + 6));
    }
  }
  else
  {
    v11 = (_QWORD *)sub_15A2DC0(v10, (__int64 *)a3, a4, 0);
  }
  *(_DWORD *)(a1 + 40) = v57;
  *(_QWORD *)(a1 + 32) = v58;
  return v11;
}
