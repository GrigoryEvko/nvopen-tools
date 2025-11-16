// Function: sub_1341570
// Address: 0x1341570
//
_QWORD *__fastcall sub_1341570(__int64 a1, __int64 a2, unsigned __int64 *a3, unsigned int a4)
{
  unsigned __int64 v8; // rax
  _QWORD *v9; // rdx
  unsigned __int64 v10; // r15
  __int64 v11; // rax
  unsigned __int64 v12; // rcx
  unsigned __int64 v13; // r15
  unsigned __int64 *v14; // rax
  unsigned __int64 v15; // rsi
  _QWORD *v16; // r15
  __int64 v17; // rbx
  _QWORD *result; // rax
  unsigned __int64 v19; // rcx
  unsigned __int64 v20; // rdi
  unsigned __int64 *v21; // rax
  unsigned __int64 v22; // r8
  unsigned __int64 v23; // rbx
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rsi
  _QWORD *v26; // r8
  unsigned int i; // edi
  __int64 v28; // r9
  _QWORD *v29; // rdi
  _QWORD *v30; // r9
  unsigned __int64 v31; // r8
  _QWORD *v32; // rsi
  unsigned int j; // r9d
  _QWORD *v34; // r10
  unsigned __int64 v35; // rax
  _QWORD *v36; // [rsp+8h] [rbp-1B8h]
  _QWORD v37[54]; // [rsp+10h] [rbp-1B0h] BYREF

  v8 = ((unsigned __int64)a4 << 17) | *a3 & 0xFFFFFFFFFFF1FFFFLL;
  v9 = (_QWORD *)(a1 + 432);
  *a3 = v8;
  if ( !a1 )
  {
    sub_130D500(v37);
    v9 = v37;
  }
  v10 = a3[1];
  v11 = (v10 >> 26) & 0xF0;
  v12 = v10 & 0xFFFFFFFFC0000000LL;
  v13 = v10 & 0xFFFFFFFFFFFFF000LL;
  v14 = (_QWORD *)((char *)v9 + v11);
  v15 = *v14;
  if ( *v14 == v12 )
  {
    v16 = (_QWORD *)(v14[1] + ((v13 >> 9) & 0x1FFFF8));
  }
  else if ( v12 == v9[32] )
  {
    v24 = v9[33];
    v9[32] = v15;
    v16 = (_QWORD *)(v24 + ((v13 >> 9) & 0x1FFFF8));
    v9[33] = v14[1];
    *v14 = v12;
    v14[1] = v24;
  }
  else
  {
    v26 = v9 + 34;
    for ( i = 1; i != 8; ++i )
    {
      if ( v12 == *v26 )
      {
        v28 = i;
        v29 = &v9[2 * i - 2];
        v30 = &v9[2 * v28];
        v31 = v30[33];
        v30[32] = v29[32];
        v16 = (_QWORD *)(v31 + ((v13 >> 9) & 0x1FFFF8));
        v30[33] = v29[33];
        v29[32] = v15;
        v29[33] = v14[1];
        *v14 = v12;
        v14[1] = v31;
        goto LABEL_5;
      }
      v26 += 2;
    }
    v36 = v9;
    v35 = sub_130D370(a1, a2, v9, v13, 1, 0);
    v9 = v36;
    v16 = (_QWORD *)v35;
  }
LABEL_5:
  v17 = 4 * a4;
  result = (_QWORD *)(a3[2] & 0xFFFFFFFFFFFFF000LL);
  if ( result == (_QWORD *)4096 )
  {
    *v16 = *v16 & 0xFFFFFFFFFFFFFFE3LL | v17;
  }
  else
  {
    v19 = (unsigned __int64)result + (a3[1] & 0xFFFFFFFFFFFFF000LL) - 4096;
    v20 = v19 & 0xFFFFFFFFC0000000LL;
    v21 = (_QWORD *)((char *)v9 + ((v19 >> 26) & 0xF0));
    v22 = *v21;
    if ( (v19 & 0xFFFFFFFFC0000000LL) == *v21 )
    {
      result = (_QWORD *)(v21[1] + ((v19 >> 9) & 0x1FFFF8));
    }
    else if ( v20 == v9[32] )
    {
      v25 = v9[33];
LABEL_16:
      v9[32] = v22;
      v9[33] = v21[1];
      *v21 = v20;
      v21[1] = v25;
      result = (_QWORD *)(v25 + ((v19 >> 9) & 0x1FFFF8));
    }
    else
    {
      v32 = v9 + 34;
      for ( j = 1; j != 8; ++j )
      {
        if ( v20 == *v32 )
        {
          v34 = &v9[2 * j];
          v9 += 2 * j - 2;
          v25 = v34[33];
          v34[32] = v9[32];
          v34[33] = v9[33];
          goto LABEL_16;
        }
        v32 += 2;
      }
      result = (_QWORD *)sub_130D370(a1, a2, v9, v19, 1, 0);
    }
    v23 = *v16 & 0xFFFFFFFFFFFFFFE3LL | v17;
    *v16 = v23;
    if ( result )
      *result = v23;
  }
  return result;
}
