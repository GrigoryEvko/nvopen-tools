// Function: sub_EDD850
// Address: 0xedd850
//
__int64 *__fastcall sub_EDD850(__int64 *a1, __int64 a2, int *a3, size_t a4, __int64 *a5)
{
  __int64 *v7; // r13
  _QWORD *v8; // r12
  __int64 v9; // rbx
  __int64 v10; // rax
  _WORD *v11; // rdx
  _QWORD *v12; // rax
  _QWORD *v13; // r10
  size_t v14; // r8
  __int64 *v15; // r9
  int v16; // r14d
  int *v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // r12
  unsigned __int64 v20; // rbx
  _QWORD *v21; // r15
  int v22; // eax
  unsigned __int64 v23; // r8
  unsigned int *v24; // rcx
  __int64 v25; // r9
  __int64 v26; // rdx
  size_t v28; // [rsp+8h] [rbp-A8h]
  __int64 *v29; // [rsp+10h] [rbp-A0h]
  __int64 v30; // [rsp+18h] [rbp-98h]
  _QWORD *v31; // [rsp+20h] [rbp-90h]
  int v32; // [rsp+2Ch] [rbp-84h]
  int v34; // [rsp+4Ch] [rbp-64h] BYREF
  _QWORD v35[12]; // [rsp+50h] [rbp-60h] BYREF

  v7 = a1;
  v8 = *(_QWORD **)(a2 + 8);
  v9 = sub_ED8690((__int64)(v8 + 4), a3, a4);
  v10 = *(_QWORD *)(v8[2] + 8 * (v9 & (*v8 - 1LL)));
  if ( !v10 || (v11 = (_WORD *)(v8[3] + v10), v12 = v11 + 1, v32 = (unsigned __int16)*v11, !*v11) )
  {
LABEL_13:
    LODWORD(v35[0]) = 13;
    sub_ED8A30(v7, (int *)v35);
    return v7;
  }
  v13 = v8 + 4;
  v14 = a4;
  v15 = a1;
  v16 = 0;
  v17 = a3;
  v18 = v9;
  while ( 1 )
  {
    v19 = v12[1];
    v20 = v12[2];
    v21 = v12 + 3;
    if ( v18 != *v12 || v14 != v19 )
      goto LABEL_11;
    if ( !v19 )
      break;
    v28 = v14;
    v29 = v15;
    v30 = v18;
    v31 = v13;
    v22 = memcmp(v12 + 3, v17, v12[1]);
    v13 = v31;
    v18 = v30;
    v15 = v29;
    v14 = v28;
    if ( !v22 )
    {
      v7 = v29;
      v23 = v20;
      v24 = (unsigned int *)((char *)v21 + v19);
      v25 = (__int64)v21;
      goto LABEL_9;
    }
LABEL_11:
    v12 = (_QWORD *)((char *)v21 + v19 + v20);
    if ( v32 == ++v16 )
    {
      v7 = v15;
      goto LABEL_13;
    }
  }
  v7 = v15;
  v23 = v12[2];
  v25 = (__int64)(v12 + 3);
  v24 = (unsigned int *)(v12 + 3);
LABEL_9:
  v35[0] = v25;
  v35[1] = v19;
  *a5 = sub_EDD200(v13, v25, v19, v24, v23);
  a5[1] = v26;
  if ( v26 )
  {
    *v7 = 1;
  }
  else
  {
    v34 = 9;
    sub_ED89C0(v7, &v34, "profile data is empty");
  }
  return v7;
}
