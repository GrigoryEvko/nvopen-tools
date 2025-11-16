// Function: sub_317E210
// Address: 0x317e210
//
void __fastcall sub_317E210(_QWORD *a1, __int64 *a2, int *a3, size_t a4)
{
  __int64 v6; // rbx
  _QWORD *v8; // r8
  size_t v9; // rdx
  int *v10; // rbx
  int *v11; // rax
  bool v12; // al
  int *v13; // rcx
  __int64 v14; // rdi
  __int64 v15; // rsi
  __int64 v16; // rcx
  int *v17; // rdi
  int *v18; // rax
  unsigned __int64 v19; // r15
  unsigned __int64 v20; // r12
  unsigned __int64 v21; // r14
  _QWORD *v22; // rdi
  int *v23; // [rsp+0h] [rbp-F0h]
  int *v24; // [rsp+8h] [rbp-E8h]
  size_t v25[2]; // [rsp+10h] [rbp-E0h] BYREF
  int v26[52]; // [rsp+20h] [rbp-D0h] BYREF

  v6 = *a2;
  if ( a3 )
  {
    sub_C7D030(v26);
    sub_C7D280(v26, a3, a4);
    sub_C7D290(v26, v25);
    a4 = v25[0];
  }
  v8 = (_QWORD *)a1[2];
  v23 = (int *)(a1 + 1);
  v9 = a4 + 33 * v6;
  if ( v8 )
  {
    v24 = (int *)(a1 + 1);
    v10 = (int *)a1[2];
    while ( 1 )
    {
      while ( v9 > *((_QWORD *)v10 + 4) )
      {
        v10 = (int *)*((_QWORD *)v10 + 3);
        if ( !v10 )
          goto LABEL_9;
      }
      v11 = (int *)*((_QWORD *)v10 + 2);
      if ( v9 >= *((_QWORD *)v10 + 4) )
        break;
      v24 = v10;
      v10 = (int *)*((_QWORD *)v10 + 2);
      if ( !v11 )
      {
LABEL_9:
        v12 = v23 == v24;
        goto LABEL_10;
      }
    }
    v13 = (int *)*((_QWORD *)v10 + 3);
    if ( v13 )
    {
      do
      {
        while ( 1 )
        {
          v14 = *((_QWORD *)v13 + 2);
          v15 = *((_QWORD *)v13 + 3);
          if ( v9 < *((_QWORD *)v13 + 4) )
            break;
          v13 = (int *)*((_QWORD *)v13 + 3);
          if ( !v15 )
            goto LABEL_19;
        }
        v24 = v13;
        v13 = (int *)*((_QWORD *)v13 + 2);
      }
      while ( v14 );
    }
LABEL_19:
    while ( v11 )
    {
      while ( 1 )
      {
        v16 = *((_QWORD *)v11 + 3);
        if ( v9 <= *((_QWORD *)v11 + 4) )
          break;
        v11 = (int *)*((_QWORD *)v11 + 3);
        if ( !v16 )
          goto LABEL_22;
      }
      v10 = v11;
      v11 = (int *)*((_QWORD *)v11 + 2);
    }
LABEL_22:
    if ( (int *)a1[3] == v10 && v23 == v24 )
      goto LABEL_12;
    for ( ; v10 != v24; --a1[5] )
    {
      v17 = v10;
      v10 = (int *)sub_220EF30((__int64)v10);
      v18 = sub_220F330(v17, v23);
      v19 = *((_QWORD *)v18 + 7);
      v20 = (unsigned __int64)v18;
      while ( v19 )
      {
        v21 = v19;
        sub_317D930(*(_QWORD **)(v19 + 24));
        v22 = *(_QWORD **)(v19 + 56);
        v19 = *(_QWORD *)(v19 + 16);
        sub_317D930(v22);
        j_j___libc_free_0(v21);
      }
      j_j___libc_free_0(v20);
    }
  }
  else
  {
    v24 = (int *)(a1 + 1);
    v12 = 1;
LABEL_10:
    if ( (int *)a1[3] == v24 && v12 )
    {
LABEL_12:
      sub_317D930(v8);
      a1[2] = 0;
      a1[5] = 0;
      a1[3] = v23;
      a1[4] = v23;
    }
  }
}
