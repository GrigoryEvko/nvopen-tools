// Function: sub_1217330
// Address: 0x1217330
//
_QWORD *__fastcall sub_1217330(_QWORD *a1, _QWORD *a2, size_t a3)
{
  _QWORD *v3; // r15
  char *v4; // rax
  __int64 v5; // rbx
  char *v6; // r13
  size_t v7; // rdx
  size_t v8; // r14
  size_t v9; // r12
  size_t v10; // rdx
  int v11; // eax
  size_t v12; // rbx
  size_t v13; // rdx
  int v14; // eax
  _QWORD *v15; // rax
  _QWORD *v16; // rsi
  __int64 v17; // rax
  _QWORD *v18; // rdx
  _QWORD *v19; // r12
  _BOOL8 v20; // rdi
  _QWORD *v22; // rdi
  size_t v23; // rbx
  size_t v24; // rdx
  unsigned int v25; // edi
  _QWORD *v27; // [rsp+8h] [rbp-38h]

  v3 = a1 + 33;
  v4 = sub_C94910((__int64)(a1 + 21), a2, a3);
  v5 = a1[34];
  v27 = a1 + 33;
  v6 = v4;
  v8 = v7;
  if ( !v5 )
  {
    v3 = a1 + 33;
    goto LABEL_19;
  }
  do
  {
    while ( 1 )
    {
      v9 = *(_QWORD *)(v5 + 40);
      v10 = v9;
      if ( v8 <= v9 )
        v10 = v8;
      if ( v10 )
      {
        v11 = memcmp(*(const void **)(v5 + 32), v6, v10);
        if ( v11 )
          break;
      }
      if ( v8 != v9 && v8 > v9 )
      {
        v5 = *(_QWORD *)(v5 + 24);
        goto LABEL_11;
      }
LABEL_3:
      v3 = (_QWORD *)v5;
      v5 = *(_QWORD *)(v5 + 16);
      if ( !v5 )
        goto LABEL_12;
    }
    if ( v11 >= 0 )
      goto LABEL_3;
    v5 = *(_QWORD *)(v5 + 24);
LABEL_11:
    ;
  }
  while ( v5 );
LABEL_12:
  if ( v27 == v3 )
  {
LABEL_19:
    v15 = (_QWORD *)sub_22077B0(72);
    v16 = v3;
    v15[4] = v6;
    v3 = v15;
    v15[5] = v8;
    v15[6] = 0;
    v15[7] = 0;
    v15[8] = 0;
    v17 = sub_9D5590(a1 + 32, v16, (__int64)(v15 + 4));
    v19 = v18;
    if ( v18 )
    {
      if ( v17 || v27 == v18 )
      {
        v20 = 1;
      }
      else
      {
        v23 = v18[5];
        v24 = v8;
        if ( v23 <= v8 )
          v24 = v23;
        if ( v24 && (v25 = memcmp(v6, (const void *)v19[4], v24)) != 0 )
        {
          v20 = v25 >> 31;
        }
        else
        {
          v20 = v23 > v8;
          if ( v23 == v8 )
            v20 = 0;
        }
      }
      sub_220F040(v20, v3, v19, v27);
      ++a1[37];
    }
    else
    {
      v22 = v3;
      v3 = (_QWORD *)v17;
      j_j___libc_free_0(v22, 72);
    }
    return v3 + 6;
  }
  v12 = v3[5];
  v13 = v8;
  if ( v12 <= v8 )
    v13 = v3[5];
  if ( v13 )
  {
    v14 = memcmp(v6, (const void *)v3[4], v13);
    if ( v14 )
    {
      if ( v14 >= 0 )
        return v3 + 6;
      goto LABEL_19;
    }
  }
  if ( v12 != v8 && v12 > v8 )
    goto LABEL_19;
  return v3 + 6;
}
