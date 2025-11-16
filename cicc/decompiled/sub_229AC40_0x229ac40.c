// Function: sub_229AC40
// Address: 0x229ac40
//
unsigned __int64 *__fastcall sub_229AC40(unsigned __int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // r8
  unsigned __int64 **v9; // rcx
  unsigned __int64 *v10; // rbx
  unsigned __int64 v11; // r9
  unsigned __int64 v12; // rsi
  size_t v13; // rdx
  int v14; // eax
  unsigned __int64 *result; // rax
  _QWORD *v16; // rax
  unsigned __int64 *v17; // rbx
  __int64 *v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rsi
  char v21; // al
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // r8
  unsigned __int64 v24; // r12
  _QWORD *v25; // r13
  unsigned __int64 **v26; // rax
  size_t v27; // r15
  _QWORD *v28; // r11
  _QWORD *v29; // rsi
  unsigned __int64 v30; // rdi
  _QWORD *v31; // rcx
  unsigned __int64 v32; // rdx
  _QWORD **v33; // rax
  unsigned __int64 v34; // rdx
  unsigned __int64 **v35; // [rsp+8h] [rbp-58h]
  unsigned __int64 v36; // [rsp+10h] [rbp-50h]
  __int64 v37; // [rsp+18h] [rbp-48h]
  unsigned __int64 v38; // [rsp+20h] [rbp-40h]
  unsigned __int64 v39; // [rsp+20h] [rbp-40h]
  unsigned __int64 v40; // [rsp+20h] [rbp-40h]

  v6 = sub_22076E0(*(__int64 **)a2, *(_QWORD *)(a2 + 8), 3339675911LL);
  v7 = a1[1];
  v8 = v6;
  v37 = v6 % v7;
  v9 = *(unsigned __int64 ***)(*a1 + v37 * 8);
  if ( !v9 )
    goto LABEL_11;
  v10 = *v9;
  v11 = v6 % v7;
  v12 = (*v9)[5];
  while ( 1 )
  {
    if ( v8 == v12 )
    {
      v13 = *(_QWORD *)(a2 + 8);
      if ( v13 == v10[2] )
      {
        v38 = v11;
        if ( !v13 )
          break;
        v35 = v9;
        v36 = v8;
        v14 = memcmp(*(const void **)a2, (const void *)v10[1], v13);
        v8 = v36;
        v9 = v35;
        v11 = v38;
        if ( !v14 )
          break;
      }
    }
    if ( !*v10 )
      goto LABEL_11;
    v12 = *(_QWORD *)(*v10 + 40);
    v9 = (unsigned __int64 **)v10;
    if ( v11 != v12 % v7 )
      goto LABEL_11;
    v10 = (unsigned __int64 *)*v10;
  }
  result = *v9;
  if ( !*v9 )
  {
LABEL_11:
    v39 = v8;
    v16 = (_QWORD *)sub_22077B0(0x30u);
    v17 = v16;
    if ( v16 )
      *v16 = 0;
    v18 = *(__int64 **)a2;
    v19 = *(_QWORD *)(a2 + 8);
    v16[1] = v16 + 3;
    sub_229AAE0(v16 + 1, v18, (__int64)v18 + v19);
    v20 = a1[1];
    v21 = sub_222DA10((__int64)(a1 + 4), v20, a1[3], a3);
    v23 = v39;
    v24 = v22;
    if ( !v21 )
    {
      v25 = (_QWORD *)*a1;
LABEL_15:
      v17[5] = v23;
      v26 = (unsigned __int64 **)&v25[v37];
      if ( v25[v37] )
      {
        *v17 = **v26;
        **v26 = (unsigned __int64)v17;
      }
      else
      {
        v34 = a1[2];
        a1[2] = (unsigned __int64)v17;
        *v17 = v34;
        if ( v34 )
        {
          v25[*(_QWORD *)(v34 + 40) % a1[1]] = v17;
          v26 = (unsigned __int64 **)(*a1 + v37 * 8);
        }
        *v26 = a1 + 2;
      }
      ++a1[3];
      return v17;
    }
    if ( v22 == 1 )
    {
      v25 = a1 + 6;
      a1[6] = 0;
      v28 = a1 + 6;
    }
    else
    {
      if ( v22 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(a1 + 4, v20, v22);
      v27 = 8 * v22;
      v25 = (_QWORD *)sub_22077B0(8 * v22);
      memset(v25, 0, v27);
      v23 = v39;
      v28 = a1 + 6;
    }
    v29 = (_QWORD *)a1[2];
    a1[2] = 0;
    if ( !v29 )
    {
LABEL_28:
      if ( (_QWORD *)*a1 != v28 )
      {
        v40 = v23;
        j_j___libc_free_0(*a1);
        v23 = v40;
      }
      a1[1] = v24;
      *a1 = (unsigned __int64)v25;
      v37 = v23 % v24;
      goto LABEL_15;
    }
    v30 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v31 = v29;
        v29 = (_QWORD *)*v29;
        v32 = v31[5] % v24;
        v33 = (_QWORD **)&v25[v32];
        if ( !*v33 )
          break;
        *v31 = **v33;
        **v33 = v31;
LABEL_24:
        if ( !v29 )
          goto LABEL_28;
      }
      *v31 = a1[2];
      a1[2] = (unsigned __int64)v31;
      *v33 = a1 + 2;
      if ( !*v31 )
      {
        v30 = v32;
        goto LABEL_24;
      }
      v25[v30] = v31;
      v30 = v32;
      if ( !v29 )
        goto LABEL_28;
    }
  }
  return result;
}
