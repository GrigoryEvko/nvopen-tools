// Function: sub_3596020
// Address: 0x3596020
//
__int64 __fastcall sub_3596020(unsigned __int64 *a1, unsigned int *a2)
{
  unsigned __int64 v3; // r13
  unsigned __int64 v4; // r8
  _QWORD *v5; // r9
  __int64 v6; // r14
  _QWORD *v7; // rcx
  _QWORD *v8; // rdi
  __int64 result; // rax
  unsigned __int64 *v10; // rax
  unsigned __int64 *v11; // r12
  int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rsi
  char v15; // al
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // r15
  unsigned __int64 v18; // r8
  unsigned __int64 **v19; // rax
  unsigned __int64 *v20; // rdx
  size_t v21; // r14
  void *v22; // rax
  void *v23; // rax
  unsigned __int64 *v24; // r10
  _QWORD *v25; // rsi
  unsigned __int64 v26; // rdi
  _QWORD *v27; // rcx
  unsigned __int64 v28; // rdx
  _QWORD **v29; // rax
  unsigned __int64 v30; // rdx
  unsigned __int64 v31; // [rsp+8h] [rbp-38h]

  v3 = *a2;
  v4 = a1[1];
  v5 = *(_QWORD **)(*a1 + 8 * (v3 % v4));
  v6 = 8 * (v3 % v4);
  if ( v5 )
  {
    v7 = (_QWORD *)*v5;
    if ( (_DWORD)v3 == *(_DWORD *)(*v5 + 8LL) )
    {
LABEL_6:
      result = *v5 + 12LL;
      if ( *v5 )
        return result;
    }
    else
    {
      while ( 1 )
      {
        v8 = (_QWORD *)*v7;
        if ( !*v7 )
          break;
        v5 = v7;
        if ( *a2 % v4 != *((unsigned int *)v8 + 2) % v4 )
          break;
        v7 = (_QWORD *)*v7;
        if ( (_DWORD)v3 == *((_DWORD *)v8 + 2) )
          goto LABEL_6;
      }
    }
  }
  v10 = (unsigned __int64 *)sub_22077B0(0x10u);
  v11 = v10;
  if ( v10 )
    *v10 = 0;
  v12 = *a2;
  v13 = a1[3];
  *((_DWORD *)v11 + 3) = 0;
  v14 = a1[1];
  *((_DWORD *)v11 + 2) = v12;
  v15 = sub_222DA10((__int64)(a1 + 4), v14, v13, 1);
  v17 = v16;
  if ( v15 )
  {
    if ( v16 == 1 )
    {
      v18 = (unsigned __int64)(a1 + 6);
      a1[6] = 0;
      v24 = a1 + 6;
    }
    else
    {
      if ( v16 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(a1 + 4, v14, v16);
      v21 = 8 * v16;
      v22 = (void *)sub_22077B0(8 * v16);
      v23 = memset(v22, 0, v21);
      v24 = a1 + 6;
      v18 = (unsigned __int64)v23;
    }
    v25 = (_QWORD *)a1[2];
    a1[2] = 0;
    if ( !v25 )
    {
LABEL_23:
      if ( v24 != (unsigned __int64 *)*a1 )
      {
        v31 = v18;
        j_j___libc_free_0(*a1);
        v18 = v31;
      }
      a1[1] = v17;
      *a1 = v18;
      v6 = 8 * (v3 % v17);
      v19 = (unsigned __int64 **)(v18 + v6);
      v20 = *(unsigned __int64 **)(v18 + v6);
      if ( v20 )
        goto LABEL_11;
LABEL_26:
      v30 = a1[2];
      a1[2] = (unsigned __int64)v11;
      *v11 = v30;
      if ( v30 )
      {
        *(_QWORD *)(v18 + 8 * (*(unsigned int *)(v30 + 8) % a1[1])) = v11;
        v19 = (unsigned __int64 **)(v6 + *a1);
      }
      *v19 = a1 + 2;
      goto LABEL_12;
    }
    v26 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v27 = v25;
        v25 = (_QWORD *)*v25;
        v28 = *((unsigned int *)v27 + 2) % v17;
        v29 = (_QWORD **)(v18 + 8 * v28);
        if ( !*v29 )
          break;
        *v27 = **v29;
        **v29 = v27;
LABEL_19:
        if ( !v25 )
          goto LABEL_23;
      }
      *v27 = a1[2];
      a1[2] = (unsigned __int64)v27;
      *v29 = a1 + 2;
      if ( !*v27 )
      {
        v26 = v28;
        goto LABEL_19;
      }
      *(_QWORD *)(v18 + 8 * v26) = v27;
      v26 = v28;
      if ( !v25 )
        goto LABEL_23;
    }
  }
  v18 = *a1;
  v19 = (unsigned __int64 **)(*a1 + v6);
  v20 = *v19;
  if ( !*v19 )
    goto LABEL_26;
LABEL_11:
  *v11 = *v20;
  **v19 = (unsigned __int64)v11;
LABEL_12:
  ++a1[3];
  return (__int64)v11 + 12;
}
