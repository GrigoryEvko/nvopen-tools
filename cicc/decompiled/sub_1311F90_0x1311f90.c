// Function: sub_1311F90
// Address: 0x1311f90
//
unsigned __int64 __fastcall sub_1311F90(__int64 a1)
{
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // r13
  __int64 v5; // rsi
  unsigned __int64 v6; // r13
  _QWORD *v7; // r11
  unsigned __int64 v8; // r14
  __int64 v9; // rbx
  unsigned __int64 *v10; // rcx
  __int64 v11; // rax
  _QWORD *v12; // rax
  _QWORD *v13; // r11
  unsigned __int64 v14; // r15
  unsigned __int64 *v15; // rbx
  unsigned __int64 v16; // rax
  _QWORD *v17; // rax
  __int64 v18; // r15
  _QWORD *v19; // r14
  unsigned __int64 v20; // rbx
  __int64 v21; // rcx
  __int64 v23; // rcx
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rdx
  __int64 v26; // rax
  _QWORD *v27; // rsi
  __int64 v28; // rdx
  unsigned __int64 v29; // rcx
  _QWORD *v30; // rdx
  unsigned int j; // ecx
  _QWORD *v32; // rsi
  _QWORD *v33; // rdx
  unsigned int i; // esi
  _QWORD *v35; // rdi
  __int64 v36; // [rsp+8h] [rbp-1B8h]
  __int64 v37; // [rsp+8h] [rbp-1B8h]
  _QWORD v38[54]; // [rsp+10h] [rbp-1B0h] BYREF

  v2 = ((_DWORD)qword_4F96AC8 + 1943) & 0xFFFFFFF8;
  if ( v2 <= 0x3800 && (unsigned __int64)qword_4F96AC0 <= 0x1000 )
  {
    v3 = -qword_4F96AC0 & (qword_4F96AC0 + v2 - 1);
    if ( v3 <= 0x1000 )
    {
      v4 = qword_505FA40[byte_5060800[(v3 + 7) >> 3]];
      goto LABEL_5;
    }
    if ( v3 <= 0x7000000000000000LL )
    {
      _BitScanReverse64(&v29, 2 * v3 - 1);
      v4 = -(1LL << ((unsigned __int8)v29 - 3)) & (v3 + (1LL << ((unsigned __int8)v29 - 3)) - 1);
LABEL_5:
      if ( v4 <= 0x3FFF )
        goto LABEL_9;
      goto LABEL_6;
    }
LABEL_28:
    v4 = 0;
    v5 = qword_50579C0[0];
    if ( qword_50579C0[0] )
      goto LABEL_10;
    goto LABEL_29;
  }
  if ( (unsigned __int64)qword_4F96AC0 > 0x7000000000000000LL )
    goto LABEL_28;
  if ( v2 > 0x4000 )
  {
    _BitScanReverse64((unsigned __int64 *)&v23, 2 * v2 - 1);
    if ( (unsigned __int64)(int)v23 < 7 )
      LOBYTE(v23) = 7;
    v4 = -(1LL << ((unsigned __int8)v23 - 3)) & (v2 + (1LL << ((unsigned __int8)v23 - 3)) - 1);
    if ( v2 <= v4 )
      goto LABEL_7;
    goto LABEL_28;
  }
LABEL_6:
  v4 = 0x4000;
LABEL_7:
  if ( __CFADD__(v4, *(_QWORD *)&dword_50607C0 + ((qword_4F96AC0 + 4095) & 0xFFFFFFFFFFFFF000LL) - 4096) )
    v4 = 0;
LABEL_9:
  v5 = qword_50579C0[0];
  if ( qword_50579C0[0] )
    goto LABEL_10;
LABEL_29:
  v5 = sub_1300B80(0, v5, (__int64)&off_49E8000);
LABEL_10:
  v6 = sub_1318040(a1, v5, v4, qword_4F96AC0, 1, 0);
  if ( !v6 )
    return 0;
  v7 = (_QWORD *)(a1 + 432);
  if ( !a1 )
  {
    sub_130D500(v38);
    v7 = v38;
  }
  v8 = v6 & 0xFFFFFFFFC0000000LL;
  v9 = (v6 >> 26) & 0xF0;
  v10 = (_QWORD *)((char *)v7 + v9);
  v11 = *(_QWORD *)((char *)v7 + v9);
  if ( (v6 & 0xFFFFFFFFC0000000LL) == v11 )
  {
    v12 = (_QWORD *)(v10[1] + ((v6 >> 9) & 0x1FFFF8));
  }
  else if ( v8 == v7[32] )
  {
    v24 = v7[33];
LABEL_33:
    v7[32] = v11;
    v7[33] = v10[1];
    *v10 = v8;
    v10[1] = v24;
    v12 = (_QWORD *)(v24 + ((v6 >> 9) & 0x1FFFF8));
  }
  else
  {
    v33 = v7 + 34;
    for ( i = 1; i != 8; ++i )
    {
      if ( v8 == *v33 )
      {
        v35 = &v7[2 * i];
        v7 += 2 * i - 2;
        v24 = v35[33];
        v35[32] = v7[32];
        v35[33] = v7[33];
        goto LABEL_33;
      }
      v33 += 2;
    }
    v12 = (_QWORD *)sub_130D370(a1, (__int64)&unk_5060AE0, v7, v6, 1, 0);
  }
  v13 = (_QWORD *)(a1 + 432);
  v14 = qword_505FA40[HIWORD(*v12)];
  if ( !a1 )
  {
    sub_130D500(v38);
    v13 = v38;
  }
  v15 = (_QWORD *)((char *)v13 + v9);
  v16 = *v15;
  if ( v8 == *v15 )
  {
    v17 = (_QWORD *)(v15[1] + ((v6 >> 9) & 0x1FFFF8));
  }
  else if ( v8 == v13[32] )
  {
    v25 = v13[33];
LABEL_36:
    v13[32] = v16;
    v13[33] = v15[1];
    *v15 = v8;
    v15[1] = v25;
    v17 = (_QWORD *)(v25 + ((v6 >> 9) & 0x1FFFF8));
  }
  else
  {
    v30 = v13 + 34;
    for ( j = 1; j != 8; ++j )
    {
      if ( v8 == *v30 )
      {
        v32 = &v13[2 * j];
        v13 += 2 * j - 2;
        v25 = v32[33];
        v32[32] = v13[32];
        v32[33] = v13[33];
        goto LABEL_36;
      }
      v30 += 2;
    }
    v17 = (_QWORD *)sub_130D370(a1, (__int64)&unk_5060AE0, v13, v6, 1, 0);
  }
  _InterlockedAdd64(
    (volatile signed __int64 *)(qword_50579C0[*(_QWORD *)(((__int64)(*v17 << 16) >> 16) & 0xFFFFFFFFFFFFFF80LL)
                                            & 0xFFFLL]
                              + 56LL),
    v14);
  v18 = v6 + qword_4F96AC8;
  v19 = (_QWORD *)(v6 + qword_4F96AC8 + 1760);
  v20 = v6 + qword_4F96AC8;
  sub_130FD00((__int64)v19, (_QWORD *)(v6 + qword_4F96AC8), v6);
  if ( *(char *)(a1 + 1) > 0 )
  {
    v21 = qword_50579C0[0];
    if ( !qword_50579C0[0] )
      v21 = sub_1300B80(a1, 0, (__int64)&off_49E8000);
  }
  else
  {
    v21 = *(_QWORD *)(a1 + 136);
    if ( !v21 )
    {
      v21 = sub_1302AE0(a1, 1);
      if ( *(_BYTE *)a1 )
      {
        v26 = *(_QWORD *)(a1 + 296);
        v27 = (_QWORD *)(a1 + 256);
        v28 = a1 + 856;
        if ( v26 )
        {
          if ( v21 != v26 )
          {
            v36 = v21;
            sub_1311F50(a1, v27, v28, v21);
            v21 = v36;
          }
        }
        else
        {
          v37 = v21;
          sub_13114E0(a1, v27, v28, v21);
          v21 = v37;
        }
      }
    }
  }
  sub_13114E0(a1, v19, v18, v21);
  return v20;
}
