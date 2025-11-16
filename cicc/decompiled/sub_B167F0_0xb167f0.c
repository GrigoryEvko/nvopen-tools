// Function: sub_B167F0
// Address: 0xb167f0
//
_BYTE *__fastcall sub_B167F0(__int64 *a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v6; // rcx
  __int64 v7; // rdi
  char *v8; // r12
  _BYTE *result; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // rax
  _BYTE *v12; // rsi
  __int64 v13; // rax
  unsigned __int64 v14; // rcx
  _BYTE *v15; // r9
  _BYTE *v16; // r10
  _BYTE *v17; // r12
  _BYTE *v18; // rsi
  _BYTE *v19; // r13
  __int64 v20; // rax
  _QWORD v21[3]; // [rsp+8h] [rbp-48h] BYREF
  char v22; // [rsp+24h] [rbp-2Ch] BYREF
  _BYTE v23[43]; // [rsp+25h] [rbp-2Bh] BYREF

  *a1 = (__int64)(a1 + 2);
  sub_B14B30(a1, a2, (__int64)&a2[a3]);
  v6 = a4;
  v7 = (__int64)(a1 + 6);
  if ( a4 < 0 )
  {
    v14 = -a4;
    v15 = v23;
    do
    {
      v16 = v15--;
      *v15 = v14 % 0xA + 48;
      result = (_BYTE *)v14;
      v14 /= 0xAu;
    }
    while ( (unsigned __int64)result > 9 );
    v17 = v16 - 2;
    *(v15 - 1) = 45;
    v18 = (_BYTE *)(v23 - (v16 - 2));
    a1[4] = v7;
    v21[0] = v18;
    v19 = v18;
    if ( (unsigned __int64)v18 > 0xF )
    {
      v20 = sub_22409D0(a1 + 4, v21, 0);
      a1[4] = v20;
      v7 = v20;
      a1[6] = v21[0];
    }
    else
    {
      if ( v18 == (_BYTE *)1 )
      {
        *((_BYTE *)a1 + 48) = 45;
LABEL_17:
        a1[5] = (__int64)v19;
        v19[v7] = 0;
        goto LABEL_6;
      }
      if ( !v18 )
        goto LABEL_17;
    }
    result = memcpy((void *)v7, v17, (size_t)v18);
    v19 = (_BYTE *)v21[0];
    v7 = a1[4];
    goto LABEL_17;
  }
  if ( a4 )
  {
    v8 = v23;
    do
    {
      *--v8 = v6 % 0xA + 48;
      v11 = v6;
      v6 /= 0xAu;
    }
    while ( v11 > 9 );
    v12 = (_BYTE *)(v23 - v8);
    a1[4] = v7;
    v21[0] = v23 - v8;
    if ( (unsigned __int64)(v23 - v8) <= 0xF )
    {
      if ( v12 == (_BYTE *)1 )
        goto LABEL_4;
      if ( !v12 )
        goto LABEL_5;
    }
    else
    {
      v13 = sub_22409D0(a1 + 4, v21, 0);
      a1[4] = v13;
      v7 = v13;
      a1[6] = v21[0];
    }
    memcpy((void *)v7, v8, v23 - v8);
    goto LABEL_5;
  }
  v22 = 48;
  v8 = &v22;
  a1[4] = v7;
  v21[0] = 1;
LABEL_4:
  *((_BYTE *)a1 + 48) = *v8;
LABEL_5:
  result = (_BYTE *)v21[0];
  v10 = a1[4];
  a1[5] = v21[0];
  result[v10] = 0;
LABEL_6:
  a1[8] = 0;
  a1[9] = 0;
  return result;
}
