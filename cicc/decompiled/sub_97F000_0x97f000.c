// Function: sub_97F000
// Address: 0x97f000
//
__int64 *__fastcall sub_97F000(__int64 *a1, __int64 a2)
{
  _BYTE *v4; // rax
  _BYTE *v5; // rdi
  size_t v6; // rdx
  const void *v7; // rsi
  _QWORD *v8; // r12
  _BYTE *v9; // rdi
  _BYTE *v10; // rax
  size_t v11; // rdx
  const void *v12; // rsi
  _BYTE *v13; // rdi
  _BYTE *v14; // rax
  const void *v15; // rsi
  size_t v16; // rbx
  _BYTE *v17; // rsi
  __int64 v18; // rdx
  _BYTE *v20; // rax
  _BYTE *v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // [rsp+8h] [rbp-198h]
  __int64 v25; // [rsp+8h] [rbp-198h]
  _QWORD v26[3]; // [rsp+10h] [rbp-190h] BYREF
  _BYTE *v27; // [rsp+28h] [rbp-178h]
  void *dest; // [rsp+30h] [rbp-170h]
  __int64 v29; // [rsp+38h] [rbp-168h]
  _QWORD *v30; // [rsp+40h] [rbp-160h]
  _QWORD v31[3]; // [rsp+50h] [rbp-150h] BYREF
  _BYTE v32[312]; // [rsp+68h] [rbp-138h] BYREF

  v29 = 0x100000000LL;
  v30 = v31;
  v31[0] = v32;
  v26[0] = &unk_49DD288;
  v31[1] = 0;
  v31[2] = 256;
  v26[1] = 2;
  v26[2] = 0;
  v27 = 0;
  dest = 0;
  sub_CB5980(v26, 0, 0, 0);
  v4 = v27;
  v5 = dest;
  v6 = *(_QWORD *)(a2 + 56);
  v7 = *(const void **)(a2 + 48);
  if ( v27 - (_BYTE *)dest < v6 )
  {
    v8 = (_QWORD *)sub_CB6200(v26, v7, v6);
    v4 = (_BYTE *)v8[3];
    v5 = (_BYTE *)v8[4];
  }
  else
  {
    v8 = v26;
    if ( v6 )
    {
      v25 = *(_QWORD *)(a2 + 56);
      memcpy(dest, v7, v6);
      v5 = (char *)dest + v25;
      dest = v5;
      if ( v27 != v5 )
        goto LABEL_4;
      goto LABEL_21;
    }
  }
  if ( v4 != v5 )
  {
LABEL_4:
    *v5 = 95;
    v9 = (_BYTE *)(v8[4] + 1LL);
    v8[4] = v9;
    goto LABEL_5;
  }
LABEL_21:
  v23 = sub_CB6200(v8, "_", 1);
  v9 = *(_BYTE **)(v23 + 32);
  v8 = (_QWORD *)v23;
LABEL_5:
  v10 = (_BYTE *)v8[3];
  v11 = *(_QWORD *)(a2 + 8);
  v12 = *(const void **)a2;
  if ( v10 - v9 < v11 )
  {
    v8 = (_QWORD *)sub_CB6200(v8, v12, v11);
    v10 = (_BYTE *)v8[3];
    v9 = (_BYTE *)v8[4];
  }
  else if ( v11 )
  {
    v24 = *(_QWORD *)(a2 + 8);
    memcpy(v9, v12, v11);
    v21 = (_BYTE *)v8[3];
    v9 = (_BYTE *)(v24 + v8[4]);
    v8[4] = v9;
    if ( v9 != v21 )
      goto LABEL_8;
    goto LABEL_19;
  }
  if ( v9 != v10 )
  {
LABEL_8:
    *v9 = 40;
    v13 = (_BYTE *)(v8[4] + 1LL);
    v8[4] = v13;
    goto LABEL_9;
  }
LABEL_19:
  v22 = sub_CB6200(v8, "(", 1);
  v13 = *(_BYTE **)(v22 + 32);
  v8 = (_QWORD *)v22;
LABEL_9:
  v14 = (_BYTE *)v8[3];
  v15 = *(const void **)(a2 + 16);
  v16 = *(_QWORD *)(a2 + 24);
  if ( v14 - v13 < v16 )
  {
    v8 = (_QWORD *)sub_CB6200(v8, v15, v16);
    v14 = (_BYTE *)v8[3];
    v13 = (_BYTE *)v8[4];
  }
  else if ( v16 )
  {
    memcpy(v13, v15, v16);
    v20 = (_BYTE *)v8[3];
    v13 = (_BYTE *)(v8[4] + v16);
    v8[4] = v13;
    if ( v13 != v20 )
      goto LABEL_12;
    goto LABEL_17;
  }
  if ( v13 != v14 )
  {
LABEL_12:
    *v13 = 41;
    ++v8[4];
    goto LABEL_13;
  }
LABEL_17:
  sub_CB6200(v8, ")", 1);
LABEL_13:
  v17 = (_BYTE *)*v30;
  v18 = v30[1];
  *a1 = (__int64)(a1 + 2);
  sub_97E470(a1, v17, (__int64)&v17[v18]);
  v26[0] = &unk_49DD388;
  sub_CB5840(v26);
  if ( (_BYTE *)v31[0] != v32 )
    _libc_free(v31[0], v17);
  return a1;
}
