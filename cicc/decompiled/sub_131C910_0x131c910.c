// Function: sub_131C910
// Address: 0x131c910
//
unsigned __int64 __fastcall sub_131C910(__int64 a1, unsigned __int64 a2)
{
  unsigned int v3; // ecx
  unsigned __int64 v4; // r12
  _QWORD *v5; // r11
  unsigned __int64 v6; // r10
  __int64 v7; // rbx
  unsigned __int64 *v8; // rcx
  __int64 v9; // rax
  _QWORD *v10; // rax
  _QWORD *v11; // r11
  unsigned __int64 v12; // r15
  unsigned __int64 *v13; // rbx
  unsigned __int64 v14; // rax
  _QWORD *v15; // rax
  char v17; // cl
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rdx
  _QWORD *v21; // rdx
  unsigned int j; // ecx
  _QWORD *v23; // rsi
  _QWORD *v24; // rdx
  unsigned int i; // esi
  _QWORD *v26; // rdi
  unsigned __int64 v27; // [rsp+0h] [rbp-1C0h]
  _QWORD v28[54]; // [rsp+10h] [rbp-1B0h] BYREF

  if ( a2 > 0x1000 )
  {
    if ( a2 > 0x7000000000000000LL )
    {
      v3 = 232;
    }
    else
    {
      v17 = 7;
      _BitScanReverse64((unsigned __int64 *)&v18, 2 * a2 - 1);
      if ( (unsigned int)v18 >= 7 )
        v17 = v18;
      if ( (unsigned int)v18 < 6 )
        LODWORD(v18) = 6;
      v3 = ((((a2 - 1) & (-1LL << (v17 - 3))) >> (v17 - 3)) & 3) + 4 * v18 - 23;
    }
  }
  else
  {
    v3 = byte_5060800[(a2 + 7) >> 3];
  }
  v4 = sub_1317CF0(a1, qword_50579C0[0], a2, v3, 0);
  if ( v4 )
  {
    v5 = (_QWORD *)(a1 + 432);
    if ( !a1 )
    {
      sub_130D500(v28);
      v5 = v28;
    }
    v6 = v4 & 0xFFFFFFFFC0000000LL;
    v7 = (v4 >> 26) & 0xF0;
    v8 = (_QWORD *)((char *)v5 + v7);
    v9 = *(_QWORD *)((char *)v5 + v7);
    if ( (v4 & 0xFFFFFFFFC0000000LL) == v9 )
    {
      v10 = (_QWORD *)(v8[1] + ((v4 >> 9) & 0x1FFFF8));
    }
    else if ( v6 == v5[32] )
    {
      v19 = v5[33];
LABEL_22:
      v5[32] = v9;
      v5[33] = v8[1];
      *v8 = v6;
      v8[1] = v19;
      v10 = (_QWORD *)(v19 + ((v4 >> 9) & 0x1FFFF8));
    }
    else
    {
      v24 = v5 + 34;
      for ( i = 1; i != 8; ++i )
      {
        if ( v6 == *v24 )
        {
          v26 = &v5[2 * i];
          v5 += 2 * i - 2;
          v19 = v26[33];
          v26[32] = v5[32];
          v26[33] = v5[33];
          goto LABEL_22;
        }
        v24 += 2;
      }
      v10 = (_QWORD *)sub_130D370(a1, (__int64)&unk_5060AE0, v5, v4, 1, 0);
      v6 = v4 & 0xFFFFFFFFC0000000LL;
    }
    v11 = (_QWORD *)(a1 + 432);
    v12 = qword_505FA40[HIWORD(*v10)];
    if ( !a1 )
    {
      v27 = v6;
      sub_130D500(v28);
      v11 = v28;
      v6 = v27;
    }
    v13 = (_QWORD *)((char *)v11 + v7);
    v14 = *v13;
    if ( v6 == *v13 )
    {
      v15 = (_QWORD *)(v13[1] + ((v4 >> 9) & 0x1FFFF8));
    }
    else if ( v6 == v11[32] )
    {
      v20 = v11[33];
LABEL_25:
      v11[32] = v14;
      v11[33] = v13[1];
      *v13 = v6;
      v13[1] = v20;
      v15 = (_QWORD *)(v20 + ((v4 >> 9) & 0x1FFFF8));
    }
    else
    {
      v21 = v11 + 34;
      for ( j = 1; j != 8; ++j )
      {
        if ( v6 == *v21 )
        {
          v23 = &v11[2 * j];
          v11 += 2 * j - 2;
          v20 = v23[33];
          v23[32] = v11[32];
          v23[33] = v11[33];
          goto LABEL_25;
        }
        v21 += 2;
      }
      v15 = (_QWORD *)sub_130D370(a1, (__int64)&unk_5060AE0, v11, v4, 1, 0);
    }
    _InterlockedAdd64(
      (volatile signed __int64 *)(qword_50579C0[*(_QWORD *)(((__int64)(*v15 << 16) >> 16) & 0xFFFFFFFFFFFFFF80LL)
                                              & 0xFFFLL]
                                + 56LL),
      v12);
  }
  return v4;
}
