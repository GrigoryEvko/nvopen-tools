// Function: sub_E8F9F0
// Address: 0xe8f9f0
//
void __fastcall sub_E8F9F0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // rsi
  __int64 v4; // rbx
  __int64 v5; // r12
  _QWORD *v6; // rax
  _QWORD **v7; // rbx
  _QWORD *v8; // r14
  unsigned int v9; // r13d
  _QWORD *v10; // rax
  __int64 v11; // r12
  _QWORD *v12; // r13
  _QWORD *v13; // rax
  unsigned int v14; // r13d
  _QWORD *v15; // rax
  _QWORD *v16; // rax
  _QWORD *v17; // rdx
  _QWORD *v18; // r13
  __int64 *v19; // r14
  __int64 *v20; // rbx
  unsigned int v21; // edx
  __int64 v22; // rax
  _QWORD *v23; // rax
  _QWORD *v24; // rax
  _QWORD *v25; // r13
  __int64 v26; // r12
  _QWORD *v27; // rax
  unsigned int v28; // r13d
  _QWORD *v29; // rax
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // r13
  _QWORD *v34; // rax
  unsigned int v35; // r13d
  __int64 v36; // rax
  __int64 v37; // rcx
  __int64 v38; // rbx
  __int64 v39; // r13
  _QWORD *v40; // rcx
  __int64 *v41; // r8
  _QWORD *v42; // rax
  __int64 v43; // rbx
  __int64 v44; // r12
  _QWORD *v45; // rax
  unsigned int v46; // ebx
  _QWORD *v47; // rax
  __int64 v48; // rax
  _QWORD *v49; // rax
  _QWORD *v50; // r12
  __int64 v51; // r13
  _QWORD *v52; // rax
  unsigned int v53; // r12d
  _QWORD *v54; // rax
  __int64 v55; // [rsp+10h] [rbp-50h]
  __int64 *v56; // [rsp+18h] [rbp-48h]
  _QWORD *v57; // [rsp+20h] [rbp-40h]
  unsigned int v58; // [rsp+20h] [rbp-40h]
  __int64 *v59; // [rsp+28h] [rbp-38h]

  v56 = a2;
  v3 = (__int64)a2 - a1;
  v55 = a3;
  if ( v3 <= 256 )
    return;
  v4 = v3;
  if ( !a3 )
  {
    v59 = v56;
    goto LABEL_48;
  }
  while ( 2 )
  {
    v5 = *(_QWORD *)(a1 + 16);
    --v55;
    v6 = *(_QWORD **)v5;
    v7 = (_QWORD **)(a1 + 16 * (v4 >> 5));
    v8 = *v7;
    if ( !*(_QWORD *)v5 )
    {
      if ( (*(_BYTE *)(v5 + 9) & 0x70) != 0x20 || *(char *)(v5 + 8) < 0 )
        BUG();
      *(_BYTE *)(v5 + 8) |= 8u;
      v6 = sub_E807D0(*(_QWORD *)(v5 + 24));
      *(_QWORD *)v5 = v6;
    }
    v9 = *(_DWORD *)(v6[1] + 36LL);
    v10 = (_QWORD *)*v8;
    if ( !*v8 )
    {
      if ( (*((_BYTE *)v8 + 9) & 0x70) != 0x20 || *((char *)v8 + 8) < 0 )
        BUG();
      *((_BYTE *)v8 + 8) |= 8u;
      v10 = sub_E807D0(v8[3]);
      *v8 = v10;
    }
    v11 = *(v56 - 2);
    if ( v9 >= *(_DWORD *)(v10[1] + 36LL) )
    {
      v33 = *(_QWORD *)(a1 + 16);
      v34 = *(_QWORD **)v33;
      if ( !*(_QWORD *)v33 )
      {
        if ( (*(_BYTE *)(v33 + 9) & 0x70) != 0x20 || *(char *)(v33 + 8) < 0 )
          BUG();
        *(_BYTE *)(v33 + 8) |= 8u;
        v34 = sub_E807D0(*(_QWORD *)(v33 + 24));
        *(_QWORD *)v33 = v34;
      }
      v35 = *(_DWORD *)(v34[1] + 36LL);
      if ( *(_QWORD *)v11 )
      {
        if ( v35 < *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v11 + 8LL) + 36LL) )
        {
LABEL_39:
          v17 = *(_QWORD **)a1;
          goto LABEL_40;
        }
      }
      else
      {
        if ( (*(_BYTE *)(v11 + 9) & 0x70) != 0x20 || *(char *)(v11 + 8) < 0 )
          BUG();
        *(_BYTE *)(v11 + 8) |= 8u;
        v49 = sub_E807D0(*(_QWORD *)(v11 + 24));
        *(_QWORD *)v11 = v49;
        if ( v35 < *(_DWORD *)(v49[1] + 36LL) )
          goto LABEL_39;
      }
      v50 = *v7;
      v51 = *(v56 - 2);
      v52 = (_QWORD *)**v7;
      if ( !v52 )
      {
        if ( (*((_BYTE *)v50 + 9) & 0x70) != 0x20 || *((char *)v50 + 8) < 0 )
          BUG();
        *((_BYTE *)v50 + 8) |= 8u;
        v52 = sub_E807D0(v50[3]);
        *v50 = v52;
      }
      v53 = *(_DWORD *)(v52[1] + 36LL);
      v54 = *(_QWORD **)v51;
      if ( !*(_QWORD *)v51 )
      {
        if ( (*(_BYTE *)(v51 + 9) & 0x70) != 0x20 || *(char *)(v51 + 8) < 0 )
          BUG();
        *(_BYTE *)(v51 + 8) |= 8u;
        v54 = sub_E807D0(*(_QWORD *)(v51 + 24));
        *(_QWORD *)v51 = v54;
      }
      v17 = *(_QWORD **)a1;
      if ( v53 >= *(_DWORD *)(v54[1] + 36LL) )
      {
        *(_QWORD *)a1 = *v7;
        *v7 = v17;
        goto LABEL_10;
      }
LABEL_58:
      *(_QWORD *)a1 = *(v56 - 2);
      *(v56 - 2) = (__int64)v17;
      v48 = *(_QWORD *)(a1 + 8);
      *(_QWORD *)(a1 + 8) = *(v56 - 1);
      *(v56 - 1) = v48;
      v17 = *(_QWORD **)(a1 + 16);
      v18 = *(_QWORD **)a1;
      goto LABEL_11;
    }
    v12 = *v7;
    v13 = (_QWORD *)**v7;
    if ( !v13 )
    {
      if ( (*((_BYTE *)v12 + 9) & 0x70) != 0x20 || *((char *)v12 + 8) < 0 )
        BUG();
      *((_BYTE *)v12 + 8) |= 8u;
      v13 = sub_E807D0(v12[3]);
      *v12 = v13;
    }
    v14 = *(_DWORD *)(v13[1] + 36LL);
    if ( !*(_QWORD *)v11 )
    {
      if ( (*(_BYTE *)(v11 + 9) & 0x70) != 0x20 || *(char *)(v11 + 8) < 0 )
        BUG();
      *(_BYTE *)(v11 + 8) |= 8u;
      v42 = sub_E807D0(*(_QWORD *)(v11 + 24));
      *(_QWORD *)v11 = v42;
      if ( v14 < *(_DWORD *)(v42[1] + 36LL) )
        goto LABEL_9;
LABEL_55:
      v43 = *(_QWORD *)(a1 + 16);
      v44 = *(v56 - 2);
      v45 = *(_QWORD **)v43;
      if ( !*(_QWORD *)v43 )
      {
        if ( (*(_BYTE *)(v43 + 9) & 0x70) != 0x20 || *(char *)(v43 + 8) < 0 )
          BUG();
        *(_BYTE *)(v43 + 8) |= 8u;
        v45 = sub_E807D0(*(_QWORD *)(v43 + 24));
        *(_QWORD *)v43 = v45;
      }
      v46 = *(_DWORD *)(v45[1] + 36LL);
      v47 = *(_QWORD **)v44;
      if ( !*(_QWORD *)v44 )
      {
        if ( (*(_BYTE *)(v44 + 9) & 0x70) != 0x20 || *(char *)(v44 + 8) < 0 )
          BUG();
        *(_BYTE *)(v44 + 8) |= 8u;
        v47 = sub_E807D0(*(_QWORD *)(v44 + 24));
        *(_QWORD *)v44 = v47;
      }
      v17 = *(_QWORD **)a1;
      if ( v46 < *(_DWORD *)(v47[1] + 36LL) )
        goto LABEL_58;
LABEL_40:
      v18 = *(_QWORD **)(a1 + 16);
      v36 = *(_QWORD *)(a1 + 8);
      *(_QWORD *)(a1 + 16) = v17;
      v37 = *(_QWORD *)(a1 + 24);
      *(_QWORD *)a1 = v18;
      *(_QWORD *)(a1 + 8) = v37;
      *(_QWORD *)(a1 + 24) = v36;
      goto LABEL_11;
    }
    if ( v14 >= *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v11 + 8LL) + 36LL) )
      goto LABEL_55;
LABEL_9:
    v15 = *(_QWORD **)a1;
    *(_QWORD *)a1 = *v7;
    *v7 = v15;
LABEL_10:
    v16 = *(_QWORD **)(a1 + 8);
    *(_QWORD *)(a1 + 8) = v7[1];
    v7[1] = v16;
    v17 = *(_QWORD **)(a1 + 16);
    v18 = *(_QWORD **)a1;
LABEL_11:
    v19 = (__int64 *)(a1 + 16);
    v20 = v56;
    while ( 1 )
    {
      v59 = v19;
      if ( !*v17 )
        break;
      v21 = *(_DWORD *)(*(_QWORD *)(*v17 + 8LL) + 36LL);
      v22 = *v18;
      if ( *v18 )
        goto LABEL_13;
LABEL_19:
      v58 = v21;
      if ( (*((_BYTE *)v18 + 9) & 0x70) != 0x20 || *((char *)v18 + 8) < 0 )
        BUG();
      *((_BYTE *)v18 + 8) |= 8u;
      v24 = sub_E807D0(v18[3]);
      *v18 = v24;
      if ( v58 >= *(_DWORD *)(v24[1] + 36LL) )
        goto LABEL_22;
LABEL_14:
      v18 = *(_QWORD **)a1;
      v17 = (_QWORD *)v19[2];
      v19 += 2;
    }
    if ( (*((_BYTE *)v17 + 9) & 0x70) != 0x20 || *((char *)v17 + 8) < 0 )
      BUG();
    *((_BYTE *)v17 + 8) |= 8u;
    v57 = v17;
    v23 = sub_E807D0(v17[3]);
    *v57 = v23;
    v21 = *(_DWORD *)(v23[1] + 36LL);
    v22 = *v18;
    if ( !*v18 )
      goto LABEL_19;
LABEL_13:
    if ( v21 < *(_DWORD *)(*(_QWORD *)(v22 + 8) + 36LL) )
      goto LABEL_14;
    do
    {
LABEL_22:
      v25 = *(_QWORD **)a1;
      v20 -= 2;
      v26 = *v20;
      v27 = **(_QWORD ***)a1;
      if ( !v27 )
      {
        if ( (*((_BYTE *)v25 + 9) & 0x70) != 0x20 || *((char *)v25 + 8) < 0 )
          BUG();
        *((_BYTE *)v25 + 8) |= 8u;
        v27 = sub_E807D0(v25[3]);
        *v25 = v27;
      }
      v28 = *(_DWORD *)(v27[1] + 36LL);
      v29 = *(_QWORD **)v26;
      if ( !*(_QWORD *)v26 )
      {
        if ( (*(_BYTE *)(v26 + 9) & 0x70) != 0x20 || *(char *)(v26 + 8) < 0 )
          BUG();
        *(_BYTE *)(v26 + 8) |= 8u;
        v29 = sub_E807D0(*(_QWORD *)(v26 + 24));
        *(_QWORD *)v26 = v29;
      }
    }
    while ( v28 < *(_DWORD *)(v29[1] + 36LL) );
    if ( v19 < v20 )
    {
      v30 = *v19;
      *v19 = *v20;
      v31 = v20[1];
      *v20 = v30;
      v32 = v19[1];
      v19[1] = v31;
      v20[1] = v32;
      goto LABEL_14;
    }
    v4 = (__int64)v19 - a1;
    sub_E8F9F0(v19, v56, v55);
    if ( (__int64)v19 - a1 > 256 )
    {
      if ( v55 )
      {
        v56 = v19;
        continue;
      }
LABEL_48:
      v38 = v4 >> 4;
      v39 = (v38 - 2) >> 1;
      sub_E8F6A0(a1, v39, v38, *(_QWORD **)(a1 + 16 * v39), *(__int64 **)(a1 + 16 * v39 + 8));
      do
      {
        --v39;
        sub_E8F6A0(a1, v39, v38, *(_QWORD **)(a1 + 16 * v39), *(__int64 **)(a1 + 16 * v39 + 8));
      }
      while ( v39 );
      do
      {
        v59 -= 2;
        v40 = (_QWORD *)*v59;
        v41 = (__int64 *)v59[1];
        *v59 = *(_QWORD *)a1;
        v59[1] = *(_QWORD *)(a1 + 8);
        sub_E8F6A0(a1, 0, ((__int64)v59 - a1) >> 4, v40, v41);
      }
      while ( (__int64)v59 - a1 > 16 );
    }
    break;
  }
}
