// Function: sub_317EE30
// Address: 0x317ee30
//
__int64 __fastcall sub_317EE30(unsigned __int64 *a1, unsigned __int64 *a2)
{
  unsigned __int64 v4; // r14
  unsigned __int64 v5; // r8
  unsigned __int64 v6; // rdx
  _QWORD *v7; // r9
  __int64 v8; // r12
  _QWORD *v9; // rsi
  _QWORD *v10; // rdi
  __int64 result; // rax
  unsigned __int64 *v12; // rax
  unsigned __int64 *v13; // r13
  unsigned __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rsi
  char v17; // al
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // r15
  unsigned __int64 v20; // r8
  unsigned __int64 **v21; // rax
  unsigned __int64 *v22; // rdx
  size_t v23; // r12
  void *v24; // rax
  void *v25; // rax
  unsigned __int64 *v26; // r10
  _QWORD *v27; // rsi
  unsigned __int64 v28; // rdi
  _QWORD *v29; // rcx
  unsigned __int64 v30; // rdx
  _QWORD **v31; // rax
  unsigned __int64 v32; // rdx
  unsigned __int64 v33; // [rsp+8h] [rbp-38h]

  v4 = *a2;
  v5 = a1[1];
  v6 = *a2 % v5;
  v7 = *(_QWORD **)(*a1 + 8 * v6);
  v8 = 8 * v6;
  if ( v7 )
  {
    v9 = (_QWORD *)*v7;
    if ( *(_QWORD *)(*v7 + 8LL) == v4 )
    {
LABEL_6:
      result = *v7 + 16LL;
      if ( *v7 )
        return result;
    }
    else
    {
      while ( 1 )
      {
        v10 = (_QWORD *)*v9;
        if ( !*v9 )
          break;
        v7 = v9;
        if ( v6 != v10[1] % v5 )
          break;
        v9 = (_QWORD *)*v9;
        if ( v10[1] == v4 )
          goto LABEL_6;
      }
    }
  }
  v12 = (unsigned __int64 *)sub_22077B0(0x18u);
  v13 = v12;
  if ( v12 )
    *v12 = 0;
  v14 = *a2;
  v15 = a1[3];
  v16 = a1[1];
  v13[2] = 0;
  v13[1] = v14;
  v17 = sub_222DA10((__int64)(a1 + 4), v16, v15, 1);
  v19 = v18;
  if ( v17 )
  {
    if ( v18 == 1 )
    {
      v20 = (unsigned __int64)(a1 + 6);
      a1[6] = 0;
      v26 = a1 + 6;
    }
    else
    {
      if ( v18 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(a1 + 4, v16, v18);
      v23 = 8 * v18;
      v24 = (void *)sub_22077B0(8 * v18);
      v25 = memset(v24, 0, v23);
      v26 = a1 + 6;
      v20 = (unsigned __int64)v25;
    }
    v27 = (_QWORD *)a1[2];
    a1[2] = 0;
    if ( !v27 )
    {
LABEL_23:
      if ( v26 != (unsigned __int64 *)*a1 )
      {
        v33 = v20;
        j_j___libc_free_0(*a1);
        v20 = v33;
      }
      a1[1] = v19;
      *a1 = v20;
      v8 = 8 * (v4 % v19);
      v21 = (unsigned __int64 **)(v20 + v8);
      v22 = *(unsigned __int64 **)(v20 + v8);
      if ( v22 )
        goto LABEL_11;
LABEL_26:
      v32 = a1[2];
      a1[2] = (unsigned __int64)v13;
      *v13 = v32;
      if ( v32 )
      {
        *(_QWORD *)(v20 + 8 * (*(_QWORD *)(v32 + 8) % a1[1])) = v13;
        v21 = (unsigned __int64 **)(v8 + *a1);
      }
      *v21 = a1 + 2;
      goto LABEL_12;
    }
    v28 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v29 = v27;
        v27 = (_QWORD *)*v27;
        v30 = v29[1] % v19;
        v31 = (_QWORD **)(v20 + 8 * v30);
        if ( !*v31 )
          break;
        *v29 = **v31;
        **v31 = v29;
LABEL_19:
        if ( !v27 )
          goto LABEL_23;
      }
      *v29 = a1[2];
      a1[2] = (unsigned __int64)v29;
      *v31 = a1 + 2;
      if ( !*v29 )
      {
        v28 = v30;
        goto LABEL_19;
      }
      *(_QWORD *)(v20 + 8 * v28) = v29;
      v28 = v30;
      if ( !v27 )
        goto LABEL_23;
    }
  }
  v20 = *a1;
  v21 = (unsigned __int64 **)(*a1 + v8);
  v22 = *v21;
  if ( !*v21 )
    goto LABEL_26;
LABEL_11:
  *v13 = *v22;
  **v21 = (unsigned __int64)v13;
LABEL_12:
  ++a1[3];
  return (__int64)(v13 + 2);
}
