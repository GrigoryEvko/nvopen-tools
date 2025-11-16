// Function: sub_29F4F20
// Address: 0x29f4f20
//
__int64 __fastcall sub_29F4F20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 v6; // r13
  _QWORD *v7; // r12
  _QWORD *v8; // rbx
  const char *v9; // rax
  size_t v10; // rdx
  size_t v11; // rbx
  const char *v12; // rax
  size_t v13; // rdx
  size_t v14; // r15
  bool v15; // cc
  size_t v16; // rdx
  int v17; // eax
  _QWORD *v18; // r14
  __int64 v19; // r15
  __int64 j; // r14
  _QWORD *v21; // r13
  const char *v22; // rax
  size_t v23; // rdx
  size_t v24; // rbx
  const char *v25; // rax
  void *v26; // rdx
  void *v27; // r8
  size_t v28; // rdx
  int v29; // eax
  __int64 v31; // rcx
  _QWORD *v32; // rax
  _QWORD *v33; // r15
  __int64 v35; // [rsp+8h] [rbp-68h]
  __int64 v37; // [rsp+18h] [rbp-58h]
  const char *s2; // [rsp+30h] [rbp-40h]
  void *s2a; // [rsp+30h] [rbp-40h]
  __int64 i; // [rsp+38h] [rbp-38h]
  const char *v42; // [rsp+38h] [rbp-38h]

  v5 = a1;
  v35 = a3 & 1;
  v37 = (a3 - 1) / 2;
  if ( a2 >= v37 )
  {
    v18 = (_QWORD *)(a1 + 16 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_24;
    v19 = a2;
LABEL_27:
    if ( (a3 - 2) / 2 == v19 )
    {
      v31 = v19 + 1;
      v19 = 2 * (v19 + 1) - 1;
      v32 = (_QWORD *)(v5 + 32 * v31 - 16);
      *v18 = *v32;
      v18[1] = v32[1];
      v18 = (_QWORD *)(v5 + 16 * v19);
    }
    goto LABEL_15;
  }
  for ( i = a2; ; i = v6 )
  {
    v6 = 2 * (i + 1);
    v7 = (_QWORD *)(a1 + 32 * (i + 1));
    v9 = sub_BD5D20(*(v7 - 1));
    v11 = v10;
    s2 = v9;
    v12 = sub_BD5D20(v7[1]);
    v14 = v13;
    v15 = v13 <= v11;
    v16 = v11;
    if ( v15 )
      v16 = v14;
    if ( v16 && (v17 = memcmp(v12, s2, v16)) != 0 )
    {
      if ( v17 < 0 )
        goto LABEL_5;
    }
    else if ( v14 != v11 && v14 < v11 )
    {
LABEL_5:
      --v6;
      v7 = (_QWORD *)(a1 + 16 * v6);
    }
    v8 = (_QWORD *)(a1 + 16 * i);
    *v8 = *v7;
    v8[1] = v7[1];
    if ( v6 >= v37 )
      break;
  }
  v18 = v7;
  v19 = v6;
  v5 = a1;
  if ( !v35 )
    goto LABEL_27;
LABEL_15:
  if ( v19 > a2 )
  {
    for ( j = (v19 - 1) / 2; ; j = (j - 1) / 2 )
    {
      v21 = (_QWORD *)(v5 + 16 * j);
      v22 = sub_BD5D20(a5);
      v24 = v23;
      v42 = v22;
      v25 = sub_BD5D20(v21[1]);
      v27 = v26;
      v15 = (unsigned __int64)v26 <= v24;
      v28 = v24;
      if ( v15 )
        v28 = (size_t)v27;
      if ( v28 && (s2a = v27, v29 = memcmp(v25, v42, v28), v27 = s2a, v29) )
      {
        if ( v29 >= 0 )
          goto LABEL_23;
      }
      else if ( v27 == (void *)v24 || (unsigned __int64)v27 >= v24 )
      {
LABEL_23:
        v18 = (_QWORD *)(v5 + 16 * v19);
        goto LABEL_24;
      }
      v33 = (_QWORD *)(v5 + 16 * v19);
      *v33 = *v21;
      v33[1] = v21[1];
      v19 = j;
      if ( a2 >= j )
        break;
    }
    v18 = (_QWORD *)(v5 + 16 * j);
  }
LABEL_24:
  *v18 = a4;
  v18[1] = a5;
  return a5;
}
