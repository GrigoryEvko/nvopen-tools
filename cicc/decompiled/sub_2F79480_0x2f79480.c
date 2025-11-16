// Function: sub_2F79480
// Address: 0x2f79480
//
void __fastcall sub_2F79480(_QWORD *a1, __int64 a2, unsigned __int64 a3, __int64 a4, int a5, __int64 a6, __int64 a7)
{
  __int64 v7; // r14
  _BYTE *v8; // rsi
  _BYTE *v9; // rax
  signed __int64 v10; // r12
  signed __int64 v11; // r15
  __int64 v12; // rax
  char *v13; // rbx
  size_t v14; // r13
  char *v15; // rax
  __int64 v16; // r13
  _BYTE *v17; // rax
  unsigned __int64 v18; // r9
  __int64 v19; // rax
  char *v20; // r15
  size_t v21; // r13
  __int64 v22; // r9
  unsigned __int64 *v23; // rax
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // r12
  unsigned __int64 v26; // [rsp+18h] [rbp-78h]
  __int64 v27; // [rsp+18h] [rbp-78h]
  unsigned __int64 v28; // [rsp+20h] [rbp-70h]
  char *v29; // [rsp+20h] [rbp-70h]
  char *v30; // [rsp+28h] [rbp-68h]
  char *v31; // [rsp+30h] [rbp-60h]
  __int64 v34; // [rsp+50h] [rbp-40h]

  v7 = (__int64)a1;
  v8 = (_BYTE *)a1[9];
  v34 = a3;
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
  v30 = &v13[v11];
  if ( v8 != v9 )
  {
    a1 = v13;
    memmove(v13, v8, v14);
  }
  v15 = &v13[v14];
  v16 = *(_QWORD *)(v7 + 48);
  v31 = v15;
  v17 = *(_BYTE **)(v16 + 8);
  v8 = *(_BYTE **)v16;
  v18 = (unsigned __int64)&v17[-*(_QWORD *)v16];
  a3 = v18;
  if ( v18 )
  {
    v28 = *(_QWORD *)(v16 + 8) - *(_QWORD *)v16;
    if ( v18 <= 0x7FFFFFFFFFFFFFFCLL )
    {
      v19 = sub_22077B0(v18);
      v8 = *(_BYTE **)v16;
      a3 = v28;
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
  v29 = &v20[a3];
  if ( v17 != v8 )
  {
    v26 = v18;
    memmove(v20, v8, v21);
    v18 = v26;
  }
  v27 = v18;
  sub_2F78DC0(v7, a2);
  sub_2F74450(
    (__int64)v13,
    v10 >> 2,
    *(_QWORD *)(v7 + 72),
    v34,
    *(_QWORD *)(v7 + 16),
    v22,
    *(_QWORD *)(v7 + 392),
    (__int64)(*(_QWORD *)(v7 + 400) - *(_QWORD *)(v7 + 392)) >> 2);
  sub_2F74360((__int64)v20, v27 >> 2, **(_QWORD **)(v7 + 48), a4, a5, a7, v34);
  v23 = *(unsigned __int64 **)(v7 + 48);
  v24 = *v23;
  *v23 = (unsigned __int64)v20;
  v23[1] = (unsigned __int64)&v20[v21];
  v23[2] = (unsigned __int64)v29;
  v25 = *(_QWORD *)(v7 + 72);
  *(_QWORD *)(v7 + 72) = v13;
  *(_QWORD *)(v7 + 80) = v31;
  *(_QWORD *)(v7 + 88) = v30;
  if ( v24 )
    j_j___libc_free_0(v24);
  if ( v25 )
    j_j___libc_free_0(v25);
}
