// Function: sub_1EE7CA0
// Address: 0x1ee7ca0
//
__int64 __fastcall sub_1EE7CA0(
        _QWORD *a1,
        unsigned __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __int64 a7)
{
  __int64 v7; // r15
  _BYTE *v8; // rsi
  _BYTE *v9; // rax
  __int64 v10; // r12
  signed __int64 v11; // r14
  __int64 v12; // rax
  char *v13; // rbx
  size_t v14; // r13
  char *v15; // rax
  __int64 v16; // r13
  _BYTE *v17; // rax
  unsigned __int64 v18; // r9
  __int64 v19; // rax
  char *v20; // r14
  size_t v21; // r13
  __int64 v22; // r9
  __int64 *v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rsi
  __int64 v26; // r12
  __int64 v27; // r13
  __int64 result; // rax
  __int64 v29; // [rsp+0h] [rbp-90h]
  unsigned __int64 v30; // [rsp+18h] [rbp-78h]
  __int64 v31; // [rsp+18h] [rbp-78h]
  unsigned __int64 v32; // [rsp+20h] [rbp-70h]
  char *v33; // [rsp+20h] [rbp-70h]
  char *v34; // [rsp+28h] [rbp-68h]
  char *v35; // [rsp+30h] [rbp-60h]

  v7 = (__int64)a1;
  v8 = (_BYTE *)a1[9];
  v9 = (_BYTE *)a1[10];
  v10 = v9 - v8;
  v11 = v9 - v8;
  if ( v9 == v8 )
  {
    v14 = 0;
    v13 = 0;
  }
  else
  {
    if ( (unsigned __int64)v10 > 0x7FFFFFFFFFFFFFFCLL )
      goto LABEL_18;
    a1 = (_QWORD *)(v9 - v8);
    v12 = sub_22077B0(v10);
    v8 = *(_BYTE **)(v7 + 72);
    v13 = (char *)v12;
    v9 = *(_BYTE **)(v7 + 80);
    v10 = v9 - v8;
    v14 = v9 - v8;
  }
  v34 = &v13[v11];
  if ( v8 != v9 )
  {
    a1 = v13;
    memmove(v13, v8, v14);
  }
  v15 = &v13[v14];
  v16 = *(_QWORD *)(v7 + 48);
  v35 = v15;
  v17 = *(_BYTE **)(v16 + 8);
  v8 = *(_BYTE **)v16;
  v18 = (unsigned __int64)&v17[-*(_QWORD *)v16];
  a3 = v18;
  if ( v18 )
  {
    v32 = *(_QWORD *)(v16 + 8) - *(_QWORD *)v16;
    if ( v18 <= 0x7FFFFFFFFFFFFFFCLL )
    {
      v19 = sub_22077B0(v18);
      v8 = *(_BYTE **)v16;
      a3 = v32;
      v20 = (char *)v19;
      v17 = *(_BYTE **)(v16 + 8);
      v18 = (unsigned __int64)&v17[-*(_QWORD *)v16];
      v21 = v18;
      goto LABEL_9;
    }
LABEL_18:
    sub_4261EA(a1, v8, a3);
  }
  v21 = 0;
  v20 = 0;
LABEL_9:
  v33 = &v20[a3];
  if ( v17 != v8 )
  {
    v30 = v18;
    memmove(v20, v8, v21);
    v18 = v30;
  }
  v31 = v18;
  sub_1EE7880(v7, a2);
  sub_1EE5590(
    (__int64)v13,
    v10 >> 2,
    *(_QWORD *)(v7 + 72),
    a4,
    *(_QWORD *)(v7 + 16),
    v22,
    *(_QWORD *)(v7 + 264),
    (__int64)(*(_QWORD *)(v7 + 272) - *(_QWORD *)(v7 + 264)) >> 2);
  sub_1EE54A0((__int64)v20, v31 >> 2, **(_QWORD **)(v7 + 48), a5, a6, a7, a4);
  v23 = *(__int64 **)(v7 + 48);
  v24 = *v23;
  v25 = v23[2];
  *v23 = (__int64)v20;
  v23[1] = (__int64)&v20[v21];
  v23[2] = (__int64)v33;
  v26 = *(_QWORD *)(v7 + 72);
  v27 = *(_QWORD *)(v7 + 88);
  *(_QWORD *)(v7 + 72) = v13;
  *(_QWORD *)(v7 + 80) = v35;
  *(_QWORD *)(v7 + 88) = v34;
  result = v29;
  if ( v24 )
    result = j_j___libc_free_0(v24, v25 - v24);
  if ( v26 )
    return j_j___libc_free_0(v26, v27 - v26);
  return result;
}
