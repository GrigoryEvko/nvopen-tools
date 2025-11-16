// Function: sub_155D020
// Address: 0x155d020
//
_QWORD *__fastcall sub_155D020(__int64 *a1, _BYTE *a2, size_t a3, _BYTE *a4, size_t a5)
{
  __int64 v7; // rax
  _QWORD *v8; // rbx
  __int64 v10; // rax
  __int64 v11; // r8
  _BYTE *v12; // rdi
  size_t v13; // rdx
  void *v14; // rdi
  __int64 v15; // rdx
  _QWORD *v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rax
  size_t v19; // [rsp+8h] [rbp-E8h]
  __int64 v21; // [rsp+18h] [rbp-D8h]
  __int64 v22; // [rsp+18h] [rbp-D8h]
  __int64 v23; // [rsp+18h] [rbp-D8h]
  __int64 v24; // [rsp+18h] [rbp-D8h]
  __int64 v25; // [rsp+18h] [rbp-D8h]
  __int64 v26; // [rsp+20h] [rbp-D0h] BYREF
  size_t v27; // [rsp+28h] [rbp-C8h] BYREF
  unsigned __int64 v28[2]; // [rsp+30h] [rbp-C0h] BYREF
  _BYTE v29[176]; // [rsp+40h] [rbp-B0h] BYREF

  v19 = a5;
  v21 = *a1;
  v28[0] = (unsigned __int64)v29;
  v28[1] = 0x2000000000LL;
  sub_16BD4E0(v28);
  if ( a5 )
    sub_16BD4E0(v28);
  v22 = v21 + 200;
  v7 = sub_16BDDE0(v22, v28, &v26);
  if ( v7 )
  {
    v8 = (_QWORD *)(v7 - 8);
    goto LABEL_5;
  }
  v10 = sub_22077B0(88);
  v11 = v22;
  v8 = (_QWORD *)v10;
  if ( !v10 )
  {
    v15 = v26;
    v16 = 0;
    goto LABEL_22;
  }
  *(_QWORD *)(v10 + 8) = 0;
  v12 = (_BYTE *)(v10 + 40);
  v13 = a3;
  *(_BYTE *)(v10 + 16) = 2;
  *(_QWORD *)(v10 + 24) = v10 + 40;
  *(_QWORD *)v10 = &unk_49ECE78;
  if ( a2 )
  {
    v27 = a3;
    if ( a3 > 0xF )
    {
      v17 = sub_22409D0(v10 + 24, &v27, 0);
      v11 = v22;
      v8[3] = v17;
      v12 = (_BYTE *)v17;
      v8[5] = v27;
    }
    else
    {
      if ( a3 == 1 )
      {
        *(_BYTE *)(v10 + 40) = *a2;
LABEL_13:
        v8[4] = v13;
        v12[v13] = 0;
        goto LABEL_15;
      }
      if ( !a3 )
        goto LABEL_13;
    }
    v23 = v11;
    memcpy(v12, a2, a3);
    v13 = v27;
    v12 = (_BYTE *)v8[3];
    v11 = v23;
    goto LABEL_13;
  }
  *(_QWORD *)(v10 + 32) = 0;
  *(_BYTE *)(v10 + 40) = 0;
LABEL_15:
  v14 = v8 + 9;
  v8[7] = v8 + 9;
  if ( !a4 )
  {
    v8[8] = 0;
    *((_BYTE *)v8 + 72) = 0;
    goto LABEL_21;
  }
  v27 = a5;
  if ( a5 > 0xF )
  {
    v25 = v11;
    v18 = sub_22409D0(v8 + 7, &v27, 0);
    v11 = v25;
    v8[7] = v18;
    v14 = (void *)v18;
    v8[9] = v27;
    goto LABEL_26;
  }
  if ( a5 != 1 )
  {
    if ( !a5 )
      goto LABEL_19;
LABEL_26:
    v24 = v11;
    memcpy(v14, a4, a5);
    v14 = (void *)v8[7];
    v11 = v24;
    v19 = v27;
    goto LABEL_19;
  }
  *((_BYTE *)v8 + 72) = *a4;
LABEL_19:
  v8[8] = v19;
  *((_BYTE *)v14 + v19) = 0;
LABEL_21:
  v15 = v26;
  v16 = v8 + 1;
LABEL_22:
  sub_16BDA20(v11, v16, v15);
LABEL_5:
  if ( (_BYTE *)v28[0] != v29 )
    _libc_free(v28[0]);
  return v8;
}
