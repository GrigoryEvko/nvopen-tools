// Function: sub_2CE79A0
// Address: 0x2ce79a0
//
unsigned __int64 __fastcall sub_2CE79A0(
        unsigned __int64 *a1,
        unsigned __int64 *a2,
        _QWORD *a3,
        unsigned __int64 a4,
        unsigned __int64 a5)
{
  __int64 v7; // rdi
  __int64 v8; // rsi
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // r14
  unsigned __int64 v11; // r8
  unsigned __int64 *v12; // rsi
  _QWORD *v13; // r10
  unsigned __int64 *v14; // rax
  _QWORD *v15; // rdi
  unsigned __int64 v17; // r12
  size_t v18; // r15
  _QWORD *v19; // rcx
  bool v20; // r15
  unsigned __int64 v21; // r9
  _QWORD *v22; // rdi
  unsigned __int64 v23; // r10
  _QWORD *v24; // r8
  bool v25; // zf
  unsigned __int64 v26; // r11
  bool v27; // si
  unsigned __int64 v28; // rdx
  char *v29; // rax
  unsigned __int64 v30; // rdx
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // rdx
  unsigned __int64 *v33; // [rsp+8h] [rbp-68h]
  unsigned __int64 v34; // [rsp+18h] [rbp-58h]
  char *v35; // [rsp+20h] [rbp-50h]

  v7 = (__int64)(a1 + 4);
  v8 = *(_QWORD *)(v7 - 24);
  if ( !(unsigned __int8)sub_222DA10(v7, v8, *(_QWORD *)(v7 - 8), 1) )
  {
    v10 = a1[1];
    goto LABEL_3;
  }
  v17 = v9;
  v10 = v9;
  if ( v9 == 1 )
  {
    a1[6] = 0;
    v35 = (char *)(a1 + 6);
    v33 = a1 + 6;
  }
  else
  {
    if ( v9 > 0xFFFFFFFFFFFFFFFLL )
      sub_4261EA(v7, v8, v9);
    v18 = 8 * v9;
    v35 = (char *)sub_22077B0(8 * v9);
    memset(v35, 0, v18);
    v33 = a1 + 6;
  }
  v19 = (_QWORD *)a1[2];
  a1[2] = 0;
  if ( v19 )
  {
    v20 = 0;
    v34 = 0;
    v21 = 0;
    v22 = 0;
    while ( 1 )
    {
      v23 = v21;
      v24 = (_QWORD *)*v19;
      v25 = v19[1] % v17 == v21;
      v26 = v19[1] % v17;
      v21 = v26;
      v27 = v25 && v22 != 0;
      if ( v27 )
        break;
      if ( v20 )
      {
        if ( *v22 )
        {
          v28 = *(_QWORD *)(*v22 + 8LL) % v17;
          if ( v28 != v23 )
            *(_QWORD *)&v35[8 * v28] = v22;
        }
      }
      v29 = &v35[8 * v26];
      if ( !*(_QWORD *)v29 )
      {
        *v19 = a1[2];
        a1[2] = (unsigned __int64)v19;
        *(_QWORD *)v29 = a1 + 2;
        if ( *v19 )
        {
          v31 = v34;
          v20 = 0;
          v34 = v26;
          *(_QWORD *)&v35[8 * v31] = v19;
        }
        else
        {
          v34 = v26;
          v20 = 0;
        }
LABEL_17:
        v22 = v19;
        if ( !v24 )
          goto LABEL_26;
        goto LABEL_18;
      }
      v20 = 0;
      v22 = v19;
      *v19 = **(_QWORD **)v29;
      **(_QWORD **)v29 = v19;
      if ( !v24 )
      {
LABEL_26:
        if ( v27 )
        {
          if ( *v19 )
          {
            v30 = *(_QWORD *)(*v19 + 8LL) % v17;
            if ( v26 != v30 )
              *(_QWORD *)&v35[8 * v30] = v19;
          }
        }
        goto LABEL_30;
      }
LABEL_18:
      v19 = v24;
    }
    v20 = v25 && v22 != 0;
    *v19 = *v22;
    *v22 = v19;
    goto LABEL_17;
  }
LABEL_30:
  if ( v33 != (unsigned __int64 *)*a1 )
    j_j___libc_free_0(*a1);
  a1[1] = v17;
  *a1 = (unsigned __int64)v35;
LABEL_3:
  v11 = a4 % v10;
  if ( a2 && a2[1] == *a3 )
  {
    *(_QWORD *)a5 = *a2;
    *a2 = a5;
LABEL_39:
    if ( *(_QWORD *)a5 )
    {
      if ( *(_QWORD *)(*(_QWORD *)a5 + 8LL) != *a3 )
      {
        v32 = *(_QWORD *)(*(_QWORD *)a5 + 8LL) % a1[1];
        if ( v11 != v32 )
          *(_QWORD *)(*a1 + 8 * v32) = a5;
      }
    }
    goto LABEL_10;
  }
  v12 = *(unsigned __int64 **)(*a1 + 8 * v11);
  if ( v12 )
  {
    v13 = (_QWORD *)*v12;
    v14 = (unsigned __int64 *)*v12;
    if ( *(_QWORD *)(*v12 + 8) == *a3 )
    {
LABEL_9:
      *(_QWORD *)a5 = *v12;
      *v12 = a5;
      if ( a2 != v12 )
        goto LABEL_10;
      goto LABEL_39;
    }
    while ( 1 )
    {
      v15 = (_QWORD *)*v14;
      if ( !*v14 )
        break;
      v12 = v14;
      if ( v11 != v15[1] % v10 )
        break;
      v14 = (unsigned __int64 *)*v14;
      if ( v15[1] == *a3 )
        goto LABEL_9;
    }
    *(_QWORD *)a5 = v13;
    **(_QWORD **)(*a1 + 8 * v11) = a5;
  }
  else
  {
    *(_QWORD *)a5 = a1[2];
    a1[2] = a5;
    if ( *(_QWORD *)a5 )
      *(_QWORD *)(*a1 + 8 * (*(_QWORD *)(*(_QWORD *)a5 + 8LL) % a1[1])) = a5;
    *(_QWORD *)(*a1 + 8 * v11) = a1 + 2;
  }
LABEL_10:
  ++a1[3];
  return a5;
}
