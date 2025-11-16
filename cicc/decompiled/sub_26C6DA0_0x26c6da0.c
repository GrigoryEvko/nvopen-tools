// Function: sub_26C6DA0
// Address: 0x26c6da0
//
__int64 __fastcall sub_26C6DA0(unsigned __int64 *a1, unsigned __int64 *a2)
{
  unsigned __int64 v4; // r14
  unsigned __int64 v5; // r8
  unsigned __int64 v6; // rdx
  _QWORD *v7; // r9
  __int64 v8; // r13
  _QWORD *v9; // rsi
  _QWORD *v10; // rdi
  __int64 v11; // r8
  unsigned __int64 *v13; // rax
  unsigned __int64 *v14; // r12
  unsigned __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rsi
  char v18; // al
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // r15
  unsigned __int64 *v21; // r9
  unsigned __int64 *v22; // rax
  unsigned __int64 *v23; // rdx
  size_t v24; // r13
  void *v25; // rax
  unsigned __int64 *v26; // rax
  unsigned __int64 *v27; // r11
  _QWORD *v28; // rsi
  unsigned __int64 v29; // rdi
  _QWORD *v30; // rcx
  unsigned __int64 v31; // rdx
  unsigned __int64 *v32; // rax
  unsigned __int64 v33; // rdx
  __int64 v34; // [rsp+0h] [rbp-40h]
  unsigned __int64 *v35; // [rsp+8h] [rbp-38h]

  v4 = *a2;
  v5 = a1[1];
  v6 = *a2 % v5;
  v7 = *(_QWORD **)(*a1 + 8 * v6);
  v8 = v6;
  if ( !v7 )
    goto LABEL_8;
  v9 = (_QWORD *)*v7;
  if ( v4 != *(_QWORD *)(*v7 + 8LL) )
  {
    do
    {
      v10 = (_QWORD *)*v9;
      if ( !*v9 )
        goto LABEL_8;
      v7 = v9;
      if ( v6 != v10[1] % v5 )
        goto LABEL_8;
      v9 = (_QWORD *)*v9;
    }
    while ( v4 != v10[1] );
  }
  v11 = *v7 + 16LL;
  if ( !*v7 )
  {
LABEL_8:
    v13 = (unsigned __int64 *)sub_22077B0(0x40u);
    v14 = v13;
    if ( v13 )
      *v13 = 0;
    v15 = *a2;
    v16 = a1[3];
    v17 = a1[1];
    *((_OWORD *)v14 + 1) = 0;
    *((_OWORD *)v14 + 3) = 0;
    v14[2] = (unsigned __int64)(v14 + 2);
    v14[1] = v15;
    v14[4] = 0;
    v14[5] = 0;
    *((_DWORD *)v14 + 14) = 0;
    v18 = sub_222DA10((__int64)(a1 + 4), v17, v16, 1);
    v11 = (__int64)(v14 + 2);
    v20 = v19;
    if ( !v18 )
    {
      v21 = (unsigned __int64 *)*a1;
      v22 = (unsigned __int64 *)(*a1 + v8 * 8);
      v23 = (unsigned __int64 *)*v22;
      if ( *v22 )
      {
LABEL_12:
        *v14 = *v23;
        *(_QWORD *)*v22 = v14;
LABEL_13:
        ++a1[3];
        return v11;
      }
LABEL_27:
      v33 = a1[2];
      a1[2] = (unsigned __int64)v14;
      *v14 = v33;
      if ( v33 )
      {
        v21[*(_QWORD *)(v33 + 8) % a1[1]] = (unsigned __int64)v14;
        v22 = (unsigned __int64 *)(v8 * 8 + *a1);
      }
      *v22 = (unsigned __int64)(a1 + 2);
      goto LABEL_13;
    }
    if ( v19 == 1 )
    {
      v21 = a1 + 6;
      a1[6] = 0;
      v27 = a1 + 6;
    }
    else
    {
      if ( v19 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(a1 + 4, v17, v19);
      v24 = 8 * v19;
      v25 = (void *)sub_22077B0(8 * v19);
      v26 = (unsigned __int64 *)memset(v25, 0, v24);
      v11 = (__int64)(v14 + 2);
      v27 = a1 + 6;
      v21 = v26;
    }
    v28 = (_QWORD *)a1[2];
    a1[2] = 0;
    if ( !v28 )
    {
LABEL_24:
      if ( v27 != (unsigned __int64 *)*a1 )
      {
        v34 = v11;
        v35 = v21;
        j_j___libc_free_0(*a1);
        v11 = v34;
        v21 = v35;
      }
      a1[1] = v20;
      *a1 = (unsigned __int64)v21;
      v8 = v4 % v20;
      v22 = &v21[v8];
      v23 = (unsigned __int64 *)v21[v8];
      if ( v23 )
        goto LABEL_12;
      goto LABEL_27;
    }
    v29 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v30 = v28;
        v28 = (_QWORD *)*v28;
        v31 = v30[1] % v20;
        v32 = &v21[v31];
        if ( !*v32 )
          break;
        *v30 = *(_QWORD *)*v32;
        *(_QWORD *)*v32 = v30;
LABEL_20:
        if ( !v28 )
          goto LABEL_24;
      }
      *v30 = a1[2];
      a1[2] = (unsigned __int64)v30;
      *v32 = (unsigned __int64)(a1 + 2);
      if ( !*v30 )
      {
        v29 = v31;
        goto LABEL_20;
      }
      v21[v29] = (unsigned __int64)v30;
      v29 = v31;
      if ( !v28 )
        goto LABEL_24;
    }
  }
  return v11;
}
