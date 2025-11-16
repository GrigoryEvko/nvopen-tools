// Function: sub_1341260
// Address: 0x1341260
//
char __fastcall sub_1341260(
        __int64 a1,
        __int64 a2,
        _QWORD *a3,
        __int64 a4,
        char a5,
        char a6,
        __int64 *a7,
        unsigned __int64 *a8)
{
  char v9; // bl
  unsigned __int64 v10; // r10
  __int64 v11; // rax
  unsigned __int64 v12; // r11
  unsigned __int64 v13; // r10
  unsigned __int64 *v14; // rax
  unsigned __int64 v15; // r13
  __int64 v16; // r10
  char v17; // bl
  char result; // al
  unsigned __int64 v19; // rcx
  unsigned __int64 v20; // r10
  unsigned __int64 *v21; // rax
  unsigned __int64 v22; // r11
  unsigned __int64 v23; // rcx
  unsigned __int64 v24; // rcx
  unsigned __int64 v25; // rsi
  _QWORD *v26; // rcx
  unsigned int i; // r14d
  _QWORD *v28; // r15
  _QWORD *v29; // r14
  _QWORD *v30; // r13
  unsigned int j; // r12d
  _QWORD *v32; // r8
  unsigned __int64 v33; // rax
  char v34; // [rsp+0h] [rbp-50h]
  char v35; // [rsp+4h] [rbp-4Ch]
  _QWORD *v36; // [rsp+8h] [rbp-48h]
  unsigned __int64 v37; // [rsp+18h] [rbp-38h]

  v9 = a5;
  v10 = *(_QWORD *)(a4 + 8);
  v11 = (v10 >> 26) & 0xF0;
  v12 = v10 & 0xFFFFFFFFC0000000LL;
  v13 = v10 & 0xFFFFFFFFFFFFF000LL;
  v14 = (_QWORD *)((char *)a3 + v11);
  v15 = *v14;
  if ( v12 == *v14 )
  {
    v16 = v14[1] + ((v13 >> 9) & 0x1FFFF8);
  }
  else if ( v12 == a3[32] )
  {
    v24 = a3[33];
    a3[32] = v15;
    v16 = v24 + ((v13 >> 9) & 0x1FFFF8);
    a3[33] = v14[1];
    *v14 = v12;
    v14[1] = v24;
  }
  else
  {
    v26 = a3 + 34;
    for ( i = 1; i != 8; ++i )
    {
      if ( v12 == *v26 )
      {
        v28 = &a3[2 * i];
        v29 = &a3[2 * i - 2];
        v37 = v28[33];
        v28[32] = v29[32];
        v28[33] = v29[33];
        v29[32] = v15;
        v16 = v37 + ((v13 >> 9) & 0x1FFFF8);
        v29[33] = v14[1];
        *v14 = v12;
        v14[1] = v37;
        goto LABEL_3;
      }
      v26 += 2;
    }
    v34 = a6;
    v35 = a5;
    v36 = a3;
    v33 = sub_130D370(a1, a2, a3, v13, a5, a6);
    a6 = v34;
    a5 = v35;
    a3 = v36;
    v16 = v33;
  }
LABEL_3:
  v17 = v9 ^ 1;
  *a7 = v16;
  result = v17 & (v16 == 0);
  if ( !result )
  {
    v19 = (*(_QWORD *)(a4 + 16) & 0xFFFFFFFFFFFFF000LL) + (*(_QWORD *)(a4 + 8) & 0xFFFFFFFFFFFFF000LL) - 4096;
    v20 = v19 & 0xFFFFFFFFC0000000LL;
    v21 = (_QWORD *)((char *)a3 + ((v19 >> 26) & 0xF0));
    v22 = *v21;
    if ( (v19 & 0xFFFFFFFFC0000000LL) == *v21 )
    {
      v23 = v21[1] + ((v19 >> 9) & 0x1FFFF8);
    }
    else if ( v20 == a3[32] )
    {
      v25 = a3[33];
LABEL_12:
      a3[32] = v22;
      a3[33] = v21[1];
      v23 = v25 + ((v19 >> 9) & 0x1FFFF8);
      *v21 = v20;
      v21[1] = v25;
    }
    else
    {
      v30 = a3 + 34;
      for ( j = 1; j != 8; ++j )
      {
        if ( v20 == *v30 )
        {
          v32 = &a3[2 * j];
          a3 += 2 * j - 2;
          v25 = v32[33];
          v32[32] = a3[32];
          v32[33] = a3[33];
          goto LABEL_12;
        }
        v30 += 2;
      }
      v23 = sub_130D370(a1, a2, a3, v19, a5, a6);
    }
    *a8 = v23;
    return v17 & (v23 == 0);
  }
  return result;
}
