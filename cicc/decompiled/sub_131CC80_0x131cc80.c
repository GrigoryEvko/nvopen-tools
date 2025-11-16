// Function: sub_131CC80
// Address: 0x131cc80
//
void __fastcall sub_131CC80(__int64 a1, unsigned __int64 a2)
{
  _QWORD *v2; // r10
  unsigned __int64 v5; // r14
  __int64 v6; // rbx
  unsigned __int64 *v7; // rcx
  __int64 v8; // rax
  _QWORD *v9; // rax
  _QWORD *v10; // r10
  unsigned __int64 v11; // r15
  unsigned __int64 *v12; // rcx
  __int64 v13; // rax
  _QWORD *v14; // rax
  _QWORD *v15; // r15
  unsigned __int64 *v16; // rcx
  __int64 v17; // rax
  _QWORD *v18; // rax
  _QWORD *v19; // r15
  unsigned __int64 *v20; // rbx
  unsigned __int64 v21; // rcx
  _QWORD *v22; // r12
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rdx
  _QWORD *v26; // rdx
  unsigned int k; // esi
  _QWORD *v28; // r10
  _QWORD *v29; // rdi
  _QWORD *v30; // rdx
  unsigned int j; // esi
  _QWORD *v32; // rdi
  _QWORD *v33; // rdx
  unsigned int i; // esi
  _QWORD *v35; // rdi
  unsigned __int64 v36; // rax
  __int64 m; // rax
  int v38; // esi
  _QWORD *v39; // rdi
  _QWORD *v40; // rdx
  _QWORD v41[55]; // [rsp-1B8h] [rbp-1B8h] BYREF

  if ( a2 )
  {
    v2 = (_QWORD *)(a1 + 432);
    if ( !a1 )
    {
      sub_130D500(v41);
      v2 = v41;
    }
    v5 = a2 & 0xFFFFFFFFC0000000LL;
    v6 = (a2 >> 26) & 0xF0;
    v7 = (_QWORD *)((char *)v2 + v6);
    v8 = *(_QWORD *)((char *)v2 + v6);
    if ( (a2 & 0xFFFFFFFFC0000000LL) == v8 )
    {
      v9 = (_QWORD *)(v7[1] + ((a2 >> 9) & 0x1FFFF8));
    }
    else if ( v5 == v2[32] )
    {
      v23 = v2[33];
LABEL_24:
      v2[32] = v8;
      v2[33] = v7[1];
      *v7 = v5;
      v7[1] = v23;
      v9 = (_QWORD *)(v23 + ((a2 >> 9) & 0x1FFFF8));
    }
    else
    {
      v33 = v2 + 34;
      for ( i = 1; i != 8; ++i )
      {
        if ( v5 == *v33 )
        {
          v35 = &v2[2 * i];
          v2 += 2 * i - 2;
          v23 = v35[33];
          v35[32] = v2[32];
          v35[33] = v2[33];
          goto LABEL_24;
        }
        v33 += 2;
      }
      v9 = (_QWORD *)sub_130D370(a1, (__int64)&unk_5060AE0, v2, a2, 1, 0);
    }
    v10 = (_QWORD *)(a1 + 432);
    v11 = qword_505FA40[HIWORD(*v9)];
    if ( !a1 )
    {
      sub_130D500(v41);
      v10 = v41;
    }
    v12 = (_QWORD *)((char *)v10 + v6);
    v13 = *(_QWORD *)((char *)v10 + v6);
    if ( v5 == v13 )
    {
      v14 = (_QWORD *)(v12[1] + ((a2 >> 9) & 0x1FFFF8));
    }
    else if ( v5 == v10[32] )
    {
      v24 = v10[33];
LABEL_27:
      v10[32] = v13;
      v10[33] = v12[1];
      *v12 = v5;
      v12[1] = v24;
      v14 = (_QWORD *)(v24 + ((a2 >> 9) & 0x1FFFF8));
    }
    else
    {
      v30 = v10 + 34;
      for ( j = 1; j != 8; ++j )
      {
        if ( v5 == *v30 )
        {
          v32 = &v10[2 * j];
          v10 += 2 * j - 2;
          v24 = v32[33];
          v32[32] = v10[32];
          v32[33] = v10[33];
          goto LABEL_27;
        }
        v30 += 2;
      }
      v14 = (_QWORD *)sub_130D370(a1, (__int64)&unk_5060AE0, v10, a2, 1, 0);
    }
    _InterlockedSub64(
      (volatile signed __int64 *)(qword_50579C0[*(_QWORD *)(((__int64)(*v14 << 16) >> 16) & 0xFFFFFFFFFFFFFF80LL)
                                              & 0xFFFLL]
                                + 56LL),
      v11);
    v15 = (_QWORD *)(a1 + 432);
    if ( !a1 )
    {
      v15 = v41;
      sub_130D500(v41);
    }
    v16 = (_QWORD *)((char *)v15 + v6);
    v17 = *(_QWORD *)((char *)v15 + v6);
    if ( v5 == v17 )
    {
      v18 = (_QWORD *)(v16[1] + ((a2 >> 9) & 0x1FFFF8));
    }
    else if ( v5 == v15[32] )
    {
      v15[32] = v17;
      v25 = v15[33];
      v15[33] = v16[1];
LABEL_30:
      *v16 = v5;
      v16[1] = v25;
      v18 = (_QWORD *)(v25 + ((a2 >> 9) & 0x1FFFF8));
    }
    else
    {
      v26 = v15 + 34;
      for ( k = 1; k != 8; ++k )
      {
        if ( v5 == *v26 )
        {
          v28 = &v15[2 * k - 2];
          v29 = &v15[2 * k];
          v25 = v29[33];
          v29[32] = v28[32];
          v29[33] = v28[33];
          v28[32] = v17;
          v28[33] = v16[1];
          goto LABEL_30;
        }
        v26 += 2;
      }
      v18 = (_QWORD *)sub_130D370(a1, (__int64)&unk_5060AE0, v15, a2, 1, 0);
    }
    if ( (*v18 & 1) != 0 )
    {
      sub_1315B20(a1, a2);
    }
    else
    {
      v19 = (_QWORD *)(a1 + 432);
      if ( !a1 )
      {
        v19 = v41;
        sub_130D500(v41);
      }
      v20 = (_QWORD *)((char *)v19 + v6);
      v21 = *v20;
      if ( v5 == *v20 )
      {
        v22 = (_QWORD *)(v20[1] + ((a2 >> 9) & 0x1FFFF8));
      }
      else if ( v5 == v19[32] )
      {
        v19[32] = v21;
        v36 = v19[33];
        v19[33] = v20[1];
LABEL_42:
        *v20 = v5;
        v20[1] = v36;
        v22 = (_QWORD *)(v36 + ((a2 >> 9) & 0x1FFFF8));
      }
      else
      {
        for ( m = 1; m != 8; ++m )
        {
          v38 = m;
          if ( v5 == v19[2 * m + 32] )
          {
            v39 = &v19[2 * m];
            v36 = v39[33];
            v40 = &v19[2 * (unsigned int)(v38 - 1)];
            v39[32] = v40[32];
            v39[33] = v40[33];
            v40[32] = v21;
            v40[33] = v20[1];
            goto LABEL_42;
          }
        }
        v22 = (_QWORD *)sub_130D370(a1, (__int64)&unk_5060AE0, v19, a2, 1, 0);
      }
      sub_130A160(a1, (_QWORD *)(((__int64)(*v22 << 16) >> 16) & 0xFFFFFFFFFFFFFF80LL));
    }
  }
}
