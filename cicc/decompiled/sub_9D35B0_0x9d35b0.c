// Function: sub_9D35B0
// Address: 0x9d35b0
//
_QWORD *__fastcall sub_9D35B0(_QWORD *a1, __int64 a2)
{
  __int64 v4; // rax
  _BYTE *v5; // rsi
  __int64 v6; // rdx
  __int64 v7; // r14
  __int64 *v8; // rdi
  _QWORD *v9; // rax
  __int64 v10; // rbx
  size_t v11; // r15
  __int64 v12; // rax
  char v13; // si
  size_t v14; // r12
  const void *v15; // r14
  size_t v16; // r13
  signed __int64 v17; // rax
  size_t v18; // rdx
  _QWORD *v19; // r10
  const void *v20; // r13
  const void *v21; // r14
  int v22; // eax
  __int64 v24; // rdi
  __int64 v25; // rax
  size_t v26; // rbx
  size_t v27; // rdx
  int v28; // eax
  unsigned int v29; // edi
  const void *v30; // [rsp+8h] [rbp-58h]
  _QWORD *v31; // [rsp+10h] [rbp-50h]
  _QWORD *v33; // [rsp+20h] [rbp-40h]
  void *s1; // [rsp+28h] [rbp-38h]
  _QWORD *s1a; // [rsp+28h] [rbp-38h]
  _QWORD *s1b; // [rsp+28h] [rbp-38h]

  v4 = sub_22077B0(64);
  v5 = *(_BYTE **)a2;
  v6 = *(_QWORD *)(a2 + 8);
  v31 = (_QWORD *)v4;
  v7 = v4;
  v8 = (__int64 *)(v4 + 32);
  v4 += 48;
  *(_QWORD *)(v7 + 32) = v4;
  v30 = (const void *)v4;
  sub_9C2D70(v8, v5, (__int64)&v5[v6]);
  v9 = a1;
  v10 = a1[2];
  v33 = v9 + 1;
  if ( !v10 )
  {
    if ( v33 != (_QWORD *)a1[3] )
    {
      v10 = (__int64)(v9 + 1);
      v21 = (const void *)v31[4];
      v11 = v31[5];
      goto LABEL_27;
    }
    v19 = v9 + 1;
    goto LABEL_40;
  }
  v11 = *(_QWORD *)(v7 + 40);
  s1 = *(void **)(v7 + 32);
  while ( 1 )
  {
    v14 = *(_QWORD *)(v10 + 40);
    v15 = *(const void **)(v10 + 32);
    v16 = v14;
    if ( v11 <= v14 )
      v16 = v11;
    if ( v16 )
    {
      LODWORD(v17) = memcmp(s1, *(const void **)(v10 + 32), v16);
      if ( (_DWORD)v17 )
        goto LABEL_11;
    }
    v17 = v11 - v14;
    if ( (__int64)(v11 - v14) >= 0x80000000LL )
      break;
    if ( v17 > (__int64)0xFFFFFFFF7FFFFFFFLL )
    {
LABEL_11:
      if ( (int)v17 >= 0 )
        break;
    }
    v12 = *(_QWORD *)(v10 + 16);
    v13 = 1;
    if ( !v12 )
      goto LABEL_13;
LABEL_4:
    v10 = v12;
  }
  v12 = *(_QWORD *)(v10 + 24);
  v13 = 0;
  if ( v12 )
    goto LABEL_4;
LABEL_13:
  v18 = v16;
  v19 = (_QWORD *)v10;
  v20 = v15;
  v21 = s1;
  if ( !v13 )
    goto LABEL_14;
  if ( a1[3] == v10 )
  {
    v19 = (_QWORD *)v10;
    goto LABEL_24;
  }
LABEL_27:
  v25 = sub_220EF80(v10);
  v19 = (_QWORD *)v10;
  v14 = *(_QWORD *)(v25 + 40);
  v20 = *(const void **)(v25 + 32);
  v10 = v25;
  v18 = v14;
  if ( v11 <= v14 )
    v18 = v11;
LABEL_14:
  if ( v18 )
  {
    s1a = v19;
    v22 = memcmp(v20, v21, v18);
    v19 = s1a;
    if ( v22 )
    {
LABEL_19:
      if ( v22 < 0 )
        goto LABEL_23;
      goto LABEL_20;
    }
  }
  if ( (__int64)(v14 - v11) > 0x7FFFFFFF )
  {
LABEL_20:
    if ( v30 != v21 )
      j_j___libc_free_0(v21, v31[6] + 1LL);
    j_j___libc_free_0(v31, 64);
    return (_QWORD *)v10;
  }
  if ( (__int64)(v14 - v11) >= (__int64)0xFFFFFFFF80000000LL )
  {
    v22 = v14 - v11;
    goto LABEL_19;
  }
LABEL_23:
  if ( !v19 )
  {
    v10 = 0;
    goto LABEL_20;
  }
LABEL_24:
  v24 = 1;
  if ( v33 == v19 )
    goto LABEL_25;
  v26 = v19[5];
  v27 = v26;
  if ( v11 <= v26 )
    v27 = v11;
  if ( v27 && (s1b = v19, v28 = memcmp(v21, (const void *)v19[4], v27), v19 = s1b, (v29 = v28) != 0) )
  {
LABEL_37:
    v24 = v29 >> 31;
  }
  else
  {
    v24 = 0;
    if ( (__int64)(v11 - v26) <= 0x7FFFFFFF )
    {
      if ( (__int64)(v11 - v26) >= (__int64)0xFFFFFFFF80000000LL )
      {
        v29 = v11 - v26;
        goto LABEL_37;
      }
LABEL_40:
      v24 = 1;
    }
  }
LABEL_25:
  sub_220F040(v24, v31, v19, v33);
  ++a1[5];
  return v31;
}
