// Function: sub_13977A0
// Address: 0x13977a0
//
unsigned __int64 __fastcall sub_13977A0(_QWORD *a1, unsigned __int64 *a2)
{
  _QWORD *v2; // r8
  unsigned __int64 v3; // r13
  _QWORD *v4; // r15
  _QWORD *v5; // rax
  bool v6; // al
  __int64 v7; // rdi
  unsigned __int64 *v8; // rcx
  unsigned __int64 v9; // rdx
  _QWORD *v11; // rcx
  __int64 v12; // rdi
  __int64 v13; // rsi
  __int64 v14; // rcx
  _QWORD *v15; // rdi
  __int64 v16; // rax
  _QWORD *v17; // r14
  __int64 v18; // r12
  __int64 v19; // r8
  __int64 v20; // rbx
  __int64 v21; // rax
  _QWORD *v22; // [rsp+0h] [rbp-50h]
  _QWORD *v24; // [rsp+10h] [rbp-40h]
  __int64 v25; // [rsp+18h] [rbp-38h]

  v2 = (_QWORD *)a1[3];
  v3 = *a2;
  v22 = a1 + 2;
  if ( v2 )
  {
    v24 = a1 + 2;
    v4 = (_QWORD *)a1[3];
    while ( 1 )
    {
      while ( v3 > v4[4] )
      {
        v4 = (_QWORD *)v4[3];
        if ( !v4 )
          goto LABEL_7;
      }
      v5 = (_QWORD *)v4[2];
      if ( v3 >= v4[4] )
        break;
      v24 = v4;
      v4 = (_QWORD *)v4[2];
      if ( !v5 )
      {
LABEL_7:
        v6 = v22 == v24;
        goto LABEL_8;
      }
    }
    v11 = (_QWORD *)v4[3];
    if ( v11 )
    {
      do
      {
        while ( 1 )
        {
          v12 = v11[2];
          v13 = v11[3];
          if ( v11[4] > v3 )
            break;
          v11 = (_QWORD *)v11[3];
          if ( !v13 )
            goto LABEL_18;
        }
        v24 = v11;
        v11 = (_QWORD *)v11[2];
      }
      while ( v12 );
    }
LABEL_18:
    while ( v5 )
    {
      while ( 1 )
      {
        v14 = v5[3];
        if ( v5[4] >= v3 )
          break;
        v5 = (_QWORD *)v5[3];
        if ( !v14 )
          goto LABEL_21;
      }
      v4 = v5;
      v5 = (_QWORD *)v5[2];
    }
LABEL_21:
    if ( (_QWORD *)a1[4] == v4 && v22 == v24 )
      goto LABEL_10;
    while ( v4 != v24 )
    {
      v15 = v4;
      v4 = (_QWORD *)sub_220EF30(v4);
      v16 = sub_220F330(v15, v22);
      v17 = *(_QWORD **)(v16 + 40);
      v18 = v16;
      if ( v17 )
      {
        v19 = v17[2];
        v20 = v17[1];
        if ( v19 != v20 )
        {
          do
          {
            v21 = *(_QWORD *)(v20 + 16);
            if ( v21 != 0 && v21 != -8 && v21 != -16 )
            {
              v25 = v19;
              sub_1649B30(v20);
              v19 = v25;
            }
            v20 += 32;
          }
          while ( v19 != v20 );
          v20 = v17[1];
        }
        if ( v20 )
          j_j___libc_free_0(v20, v17[3] - v20);
        j_j___libc_free_0(v17, 40);
      }
      j_j___libc_free_0(v18, 48);
      --a1[6];
    }
  }
  else
  {
    v24 = a1 + 2;
    v6 = 1;
LABEL_8:
    if ( (_QWORD *)a1[4] == v24 && v6 )
    {
LABEL_10:
      sub_1396A40(v2);
      a1[3] = 0;
      a1[4] = v22;
      a1[5] = v22;
      a1[6] = 0;
    }
  }
  v7 = *a1 + 24LL;
  if ( !v3 )
  {
    sub_1631B90(v7, -56);
    BUG();
  }
  sub_1631B90(v7, v3);
  v8 = *(unsigned __int64 **)(v3 + 64);
  v9 = *(_QWORD *)(v3 + 56) & 0xFFFFFFFFFFFFFFF8LL;
  *v8 = v9 | *v8 & 7;
  *(_QWORD *)(v9 + 8) = v8;
  *(_QWORD *)(v3 + 64) = 0;
  *(_QWORD *)(v3 + 56) &= 7uLL;
  return v3;
}
