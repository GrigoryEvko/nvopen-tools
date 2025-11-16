// Function: sub_13124F0
// Address: 0x13124f0
//
__int64 __fastcall sub_13124F0(__int64 a1)
{
  _QWORD *v1; // r15
  _QWORD *v2; // r14
  __int64 v4; // r11
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // r10
  __int64 v7; // rsi
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  _QWORD *v10; // r11
  unsigned __int64 v11; // r10
  __int64 v12; // rbx
  unsigned __int64 *v13; // rsi
  __int64 v14; // rax
  _QWORD *v15; // rax
  unsigned __int64 v16; // r11
  _QWORD *v17; // rax
  unsigned __int64 *v18; // rbx
  unsigned __int64 v19; // rsi
  _QWORD *v20; // rax
  __int64 v21; // rcx
  __int64 v23; // rcx
  __int64 v24; // rax
  int v25; // eax
  unsigned int v26; // esi
  unsigned int v27; // eax
  __int64 v28; // r13
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // rdx
  unsigned __int64 v31; // rcx
  __int64 v32; // rax
  _QWORD *v33; // rdx
  unsigned int i; // edi
  _QWORD *v35; // r8
  _QWORD *v36; // rdx
  unsigned int j; // edi
  _QWORD *v38; // r8
  __int64 v39; // rax
  unsigned __int64 v40; // [rsp+0h] [rbp-1D0h]
  unsigned __int64 v41; // [rsp+8h] [rbp-1C8h]
  unsigned __int64 v42; // [rsp+10h] [rbp-1C0h]
  unsigned __int64 v43; // [rsp+10h] [rbp-1C0h]
  unsigned __int64 v44; // [rsp+10h] [rbp-1C0h]
  unsigned __int64 v45; // [rsp+10h] [rbp-1C0h]
  __int64 v46; // [rsp+18h] [rbp-1B8h]
  __int64 v47; // [rsp+18h] [rbp-1B8h]
  unsigned __int64 v48; // [rsp+18h] [rbp-1B8h]
  __int64 v49; // [rsp+18h] [rbp-1B8h]
  __int64 v50; // [rsp+18h] [rbp-1B8h]
  unsigned __int64 v51; // [rsp+18h] [rbp-1B8h]
  unsigned __int64 v52; // [rsp+18h] [rbp-1B8h]
  __int64 v53; // [rsp+18h] [rbp-1B8h]
  _QWORD v54[54]; // [rsp+20h] [rbp-1B0h] BYREF

  v1 = (_QWORD *)(a1 + 256);
  v2 = (_QWORD *)(a1 + 856);
  v4 = qword_4F96AC0;
  if ( (unsigned __int64)qword_4F96AC8 <= 0x3800 && (unsigned __int64)qword_4F96AC0 <= 0x1000 )
  {
    v5 = -qword_4F96AC0 & (qword_4F96AC8 + qword_4F96AC0 - 1);
    if ( v5 <= 0x1000 )
    {
      v6 = qword_505FA40[byte_5060800[(v5 + 7) >> 3]];
      goto LABEL_5;
    }
    if ( v5 <= 0x7000000000000000LL )
    {
      _BitScanReverse64(&v31, 2 * v5 - 1);
      v6 = -(1LL << ((unsigned __int8)v31 - 3)) & (v5 + (1LL << ((unsigned __int8)v31 - 3)) - 1);
LABEL_5:
      if ( v6 <= 0x3FFF )
        goto LABEL_9;
      goto LABEL_6;
    }
LABEL_29:
    v6 = 0;
    v7 = qword_50579C0[0];
    if ( qword_50579C0[0] )
      goto LABEL_10;
    goto LABEL_30;
  }
  if ( (unsigned __int64)qword_4F96AC0 > 0x7000000000000000LL )
    goto LABEL_29;
  if ( (unsigned __int64)qword_4F96AC8 > 0x4000 )
  {
    if ( (unsigned __int64)qword_4F96AC8 <= 0x7000000000000000LL )
    {
      _BitScanReverse64((unsigned __int64 *)&v23, 2 * qword_4F96AC8 - 1);
      if ( (unsigned __int64)(int)v23 < 7 )
        LOBYTE(v23) = 7;
      v6 = -(1LL << ((unsigned __int8)v23 - 3)) & (qword_4F96AC8 + (1LL << ((unsigned __int8)v23 - 3)) - 1);
      if ( qword_4F96AC8 <= v6 )
        goto LABEL_7;
    }
    goto LABEL_29;
  }
LABEL_6:
  v6 = 0x4000;
LABEL_7:
  if ( __CFADD__(v6, ((qword_4F96AC0 + 4095) & 0xFFFFFFFFFFFFF000LL) + *(_QWORD *)&dword_50607C0 - 4096) )
    v6 = 0;
LABEL_9:
  v7 = qword_50579C0[0];
  if ( qword_50579C0[0] )
    goto LABEL_10;
LABEL_30:
  v42 = v6;
  v46 = qword_4F96AC0;
  v24 = sub_1300B80(0, v7, (__int64)&off_49E8000);
  v6 = v42;
  v4 = v46;
  v7 = v24;
LABEL_10:
  v8 = sub_1318040(a1, v7, v6, v4, 1, 0);
  v9 = v8;
  if ( v8 )
  {
    v10 = (_QWORD *)(a1 + 432);
    if ( !a1 )
    {
      v48 = v8;
      sub_130D500(v54);
      v9 = v48;
      v10 = v54;
    }
    v11 = v9 & 0xFFFFFFFFC0000000LL;
    v12 = (v9 >> 26) & 0xF0;
    v13 = (_QWORD *)((char *)v10 + v12);
    v14 = *(_QWORD *)((char *)v10 + v12);
    if ( (v9 & 0xFFFFFFFFC0000000LL) == v14 )
    {
      v15 = (_QWORD *)(v13[1] + ((v9 >> 9) & 0x1FFFF8));
    }
    else if ( v11 == v10[32] )
    {
      v30 = v10[33];
LABEL_58:
      v10[32] = v14;
      v10[33] = v13[1];
      *v13 = v11;
      v13[1] = v30;
      v15 = (_QWORD *)(v30 + ((v9 >> 9) & 0x1FFFF8));
    }
    else
    {
      v33 = v10 + 34;
      for ( i = 1; i != 8; ++i )
      {
        if ( v11 == *v33 )
        {
          v35 = &v10[2 * i];
          v10 += 2 * i - 2;
          v30 = v35[33];
          v35[32] = v10[32];
          v35[33] = v10[33];
          goto LABEL_58;
        }
        v33 += 2;
      }
      v44 = v9 & 0xFFFFFFFFC0000000LL;
      v51 = v9;
      v15 = (_QWORD *)sub_130D370(a1, (__int64)&unk_5060AE0, v10, v9, 1, 0);
      v11 = v44;
      v9 = v51;
    }
    v16 = qword_505FA40[HIWORD(*v15)];
    v17 = (_QWORD *)(a1 + 432);
    if ( !a1 )
    {
      v40 = v11;
      v41 = v16;
      v43 = v9;
      sub_130D500(v54);
      v17 = v54;
      v11 = v40;
      v16 = v41;
      v9 = v43;
    }
    v18 = (_QWORD *)((char *)v17 + v12);
    v19 = *v18;
    if ( v11 == *v18 )
    {
      v20 = (_QWORD *)(v18[1] + ((v9 >> 9) & 0x1FFFF8));
    }
    else if ( v11 == v17[32] )
    {
      v29 = v17[33];
LABEL_55:
      v17[32] = v19;
      v17[33] = v18[1];
      *v18 = v11;
      v18[1] = v29;
      v20 = (_QWORD *)(v29 + ((v9 >> 9) & 0x1FFFF8));
    }
    else
    {
      v36 = v17 + 34;
      for ( j = 1; j != 8; ++j )
      {
        if ( v11 == *v36 )
        {
          v38 = &v17[2 * j];
          v17 += 2 * j - 2;
          v29 = v38[33];
          v38[32] = v17[32];
          v38[33] = v17[33];
          goto LABEL_55;
        }
        v36 += 2;
      }
      v45 = v16;
      v52 = v9;
      v20 = (_QWORD *)sub_130D370(a1, (__int64)&unk_5060AE0, v17, v9, 1, 0);
      v16 = v45;
      v9 = v52;
    }
    _InterlockedAdd64(
      (volatile signed __int64 *)(qword_50579C0[*(_QWORD *)(((__int64)(*v20 << 16) >> 16) & 0xFFFFFFFFFFFFFF80LL)
                                              & 0xFFFLL]
                                + 56LL),
      v16);
    sub_130FD00((__int64)v1, v2, v9);
    *(_QWORD *)(a1 + 296) = 0;
    if ( dword_4C6F034[0] )
    {
      v21 = qword_50579C0[0];
LABEL_21:
      sub_13114E0(a1, v1, (__int64)v2, v21);
      return 0;
    }
    if ( *(char *)(a1 + 1) > 0 )
    {
      v21 = qword_50579C0[0];
      if ( !qword_50579C0[0] )
        v21 = sub_1300B80(a1, 0, (__int64)&off_49E8000);
LABEL_50:
      if ( *(_QWORD *)(a1 + 296) )
        return 0;
      goto LABEL_21;
    }
    v21 = *(_QWORD *)(a1 + 144);
    if ( v21 )
    {
      v25 = unk_4C6F238;
      if ( unk_4C6F238 <= 2u )
        goto LABEL_21;
      goto LABEL_34;
    }
    v21 = sub_1302AE0(a1, 0);
    if ( *(_BYTE *)a1 )
    {
      v32 = *(_QWORD *)(a1 + 296);
      if ( v32 )
      {
        if ( v21 == v32 )
        {
          v25 = unk_4C6F238;
          if ( unk_4C6F238 <= 2u )
            return 0;
LABEL_34:
          v26 = dword_505F9BC;
          if ( v25 == 4 && dword_505F9BC > 1u )
            v26 = (dword_505F9BC >> 1) - (((dword_505F9BC & 1) == 0) - 1);
          if ( *(_DWORD *)(v21 + 78928) < v26 && a1 != *(_QWORD *)(v21 + 16) )
          {
            v47 = v21;
            v27 = sched_getcpu();
            v21 = v47;
            if ( unk_4C6F238 != 3 && dword_505F9BC >> 1 <= v27 )
              v27 -= dword_505F9BC >> 1;
            if ( *(_DWORD *)(v47 + 78928) != v27 )
            {
              v21 = *(_QWORD *)(a1 + 144);
              if ( v27 != *(_DWORD *)(v21 + 78928) )
              {
                v28 = qword_50579C0[v27];
                if ( !v28 )
                {
                  v53 = *(_QWORD *)(a1 + 144);
                  v39 = sub_1300B80(a1, v27, (__int64)&off_49E8000);
                  v21 = v53;
                  v28 = v39;
                }
                sub_1302A70(a1, v21, v28);
                if ( *(_BYTE *)a1 )
                  sub_1311F50(a1, v1, (__int64)v2, v28);
                v21 = *(_QWORD *)(a1 + 144);
              }
            }
            *(_QWORD *)(v21 + 16) = a1;
          }
          goto LABEL_50;
        }
        v49 = v21;
        sub_1311F50(a1, v1, (__int64)v2, v21);
        v21 = v49;
      }
      else
      {
        v50 = v21;
        sub_13114E0(a1, v1, (__int64)v2, v21);
        v21 = v50;
      }
    }
    v25 = unk_4C6F238;
    if ( unk_4C6F238 <= 2u )
      goto LABEL_50;
    goto LABEL_34;
  }
  return 1;
}
