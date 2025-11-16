// Function: sub_1311920
// Address: 0x1311920
//
__int64 __fastcall sub_1311920(__int64 a1, __int64 *a2)
{
  __int64 v3; // rbx
  __int64 v4; // r14
  unsigned __int64 v5; // r11
  _QWORD *v6; // r10
  unsigned __int64 v7; // r13
  __int64 v8; // rbx
  unsigned __int64 *v9; // rcx
  __int64 v10; // rax
  _QWORD *v11; // rax
  _QWORD *v12; // r15
  __int64 v13; // rax
  unsigned __int64 v14; // r10
  unsigned __int64 *v15; // rcx
  __int64 v16; // rax
  _QWORD *v17; // rax
  _QWORD *v18; // r10
  unsigned __int64 *v19; // rcx
  __int64 v20; // rax
  _QWORD *v21; // rax
  __int64 v22; // rcx
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rdx
  unsigned __int64 v26; // rdx
  _QWORD *v27; // rdx
  unsigned __int64 *v28; // rbx
  unsigned __int64 v29; // rcx
  _QWORD *v30; // r11
  _QWORD *v31; // rdx
  unsigned int i; // esi
  _QWORD *v33; // rdi
  unsigned __int64 v34; // rax
  _QWORD *v35; // rdx
  unsigned int k; // esi
  _QWORD *v37; // rdi
  _QWORD *v38; // rdx
  unsigned int j; // esi
  _QWORD *v40; // rdi
  __int64 m; // rax
  int v42; // esi
  _QWORD *v43; // rdi
  unsigned __int64 v44; // [rsp+0h] [rbp-1C0h]
  unsigned __int64 v45; // [rsp+0h] [rbp-1C0h]
  unsigned __int64 v46; // [rsp+0h] [rbp-1C0h]
  unsigned __int64 v47; // [rsp+0h] [rbp-1C0h]
  unsigned __int64 v48; // [rsp+8h] [rbp-1B8h]
  unsigned __int64 v49; // [rsp+8h] [rbp-1B8h]
  unsigned __int64 v50; // [rsp+8h] [rbp-1B8h]
  unsigned __int64 v51; // [rsp+8h] [rbp-1B8h]
  unsigned __int64 v52; // [rsp+8h] [rbp-1B8h]
  _QWORD v53[54]; // [rsp+10h] [rbp-1B0h] BYREF

  v3 = *a2;
  sub_1311250(a1, a2);
  v4 = *(_QWORD *)(v3 + 40);
  sub_1311780(a1, v3);
  v5 = *(_QWORD *)(v3 + 160);
  v6 = (_QWORD *)(a1 + 432);
  if ( !a1 )
  {
    v48 = *(_QWORD *)(v3 + 160);
    sub_130D500(v53);
    v5 = v48;
    v6 = v53;
  }
  v7 = v5 & 0xFFFFFFFFC0000000LL;
  v8 = (v5 >> 26) & 0xF0;
  v9 = (_QWORD *)((char *)v6 + v8);
  v10 = *(_QWORD *)((char *)v6 + v8);
  if ( (v5 & 0xFFFFFFFFC0000000LL) == v10 )
  {
    v11 = (_QWORD *)(v9[1] + ((v5 >> 9) & 0x1FFFF8));
  }
  else if ( v7 == v6[32] )
  {
    v24 = v6[33];
LABEL_19:
    v6[32] = v10;
    v6[33] = v9[1];
    *v9 = v7;
    v9[1] = v24;
    v11 = (_QWORD *)(v24 + ((v5 >> 9) & 0x1FFFF8));
  }
  else
  {
    v31 = v6 + 34;
    for ( i = 1; i != 8; ++i )
    {
      if ( v7 == *v31 )
      {
        v33 = &v6[2 * i];
        v6 += 2 * i - 2;
        v24 = v33[33];
        v33[32] = v6[32];
        v33[33] = v6[33];
        goto LABEL_19;
      }
      v31 += 2;
    }
    v50 = v5;
    v11 = (_QWORD *)sub_130D370(a1, (__int64)&unk_5060AE0, v6, v5, 1, 0);
    v5 = v50;
  }
  v12 = (_QWORD *)(a1 + 432);
  v13 = HIWORD(*v11);
  v14 = qword_505FA40[v13];
  if ( a1 )
  {
    v15 = (_QWORD *)((char *)v12 + v8);
    v16 = *(_QWORD *)((char *)v12 + v8);
    if ( v7 == v16 )
    {
LABEL_7:
      v17 = (_QWORD *)(v15[1] + ((v5 >> 9) & 0x1FFFF8));
      goto LABEL_8;
    }
  }
  else
  {
    v12 = v53;
    v44 = qword_505FA40[v13];
    v49 = v5;
    sub_130D500(v53);
    v15 = (_QWORD *)((char *)v53 + v8);
    v14 = v44;
    v5 = v49;
    v16 = *(_QWORD *)((char *)v53 + v8);
    if ( v7 == v16 )
      goto LABEL_7;
  }
  if ( v7 == v12[32] )
  {
    v25 = v12[33];
LABEL_23:
    v12[32] = v16;
    v12[33] = v15[1];
    *v15 = v7;
    v15[1] = v25;
    v17 = (_QWORD *)(v25 + ((v5 >> 9) & 0x1FFFF8));
  }
  else
  {
    v38 = v12 + 34;
    for ( j = 1; j != 8; ++j )
    {
      if ( v7 == *v38 )
      {
        v40 = &v12[2 * j];
        v12 += 2 * j - 2;
        v25 = v40[33];
        v40[32] = v12[32];
        v40[33] = v12[33];
        goto LABEL_23;
      }
      v38 += 2;
    }
    v47 = v14;
    v52 = v5;
    v17 = (_QWORD *)sub_130D370(a1, (__int64)&unk_5060AE0, v12, v5, 1, 0);
    v14 = v47;
    v5 = v52;
  }
LABEL_8:
  _InterlockedSub64(
    (volatile signed __int64 *)(qword_50579C0[*(_QWORD *)(((__int64)(*v17 << 16) >> 16) & 0xFFFFFFFFFFFFFF80LL)
                                            & 0xFFFLL]
                              + 56LL),
    v14);
  v18 = (_QWORD *)(a1 + 432);
  if ( a1 )
  {
    v19 = (_QWORD *)((char *)v18 + v8);
    v20 = *(_QWORD *)((char *)v18 + v8);
    if ( v7 == v20 )
    {
LABEL_10:
      v21 = (_QWORD *)(v19[1] + ((v5 >> 9) & 0x1FFFF8));
      goto LABEL_11;
    }
  }
  else
  {
    v45 = v5;
    sub_130D500(v53);
    v18 = v53;
    v5 = v45;
    v19 = (_QWORD *)((char *)v53 + v8);
    v20 = *(_QWORD *)((char *)v53 + v8);
    if ( v7 == v20 )
      goto LABEL_10;
  }
  if ( v7 == v18[32] )
  {
    v26 = v18[33];
LABEL_27:
    v18[32] = v20;
    v18[33] = v19[1];
    *v19 = v7;
    v19[1] = v26;
    v21 = (_QWORD *)(v26 + ((v5 >> 9) & 0x1FFFF8));
  }
  else
  {
    v35 = v18 + 34;
    for ( k = 1; k != 8; ++k )
    {
      if ( v7 == *v35 )
      {
        v37 = &v18[2 * k];
        v18 += 2 * k - 2;
        v26 = v37[33];
        v37[32] = v18[32];
        v37[33] = v18[33];
        goto LABEL_27;
      }
      v35 += 2;
    }
    v51 = v5;
    v21 = (_QWORD *)sub_130D370(a1, (__int64)&unk_5060AE0, v18, v5, 1, 0);
    v5 = v51;
  }
LABEL_11:
  if ( (*v21 & 1) != 0 )
  {
    sub_1315B20(a1, v5);
  }
  else
  {
    v27 = (_QWORD *)(a1 + 432);
    if ( !a1 )
    {
      v46 = v5;
      sub_130D500(v53);
      v27 = v53;
      v5 = v46;
    }
    v28 = (_QWORD *)((char *)v27 + v8);
    v29 = *v28;
    if ( v7 == *v28 )
    {
      v30 = (_QWORD *)(v28[1] + ((v5 >> 9) & 0x1FFFF8));
    }
    else if ( v7 == v27[32] )
    {
      v34 = v27[33];
LABEL_38:
      v27[32] = v29;
      v27[33] = v28[1];
      v30 = (_QWORD *)(v34 + ((v5 >> 9) & 0x1FFFF8));
      *v28 = v7;
      v28[1] = v34;
    }
    else
    {
      for ( m = 1; m != 8; ++m )
      {
        v42 = m;
        if ( v7 == v27[2 * m + 32] )
        {
          v43 = &v27[2 * m];
          v34 = v43[33];
          v27 += 2 * (unsigned int)(v42 - 1);
          v43[32] = v27[32];
          v43[33] = v27[33];
          goto LABEL_38;
        }
      }
      v30 = (_QWORD *)sub_130D370(a1, (__int64)&unk_5060AE0, v27, v5, 1, 0);
    }
    sub_130A160(a1, (_QWORD *)(((__int64)(*v30 << 16) >> 16) & 0xFFFFFFFFFFFFFF80LL));
  }
  sub_1315160(a1, qword_50579C0[0], 0, 0);
  if ( (unsigned int)sub_1317080(v4, 0) || (v22 = 1, unk_5260DD0) )
    v22 = 0;
  return sub_1315160(a1, v4, 0, v22);
}
