// Function: sub_3021810
// Address: 0x3021810
//
void __fastcall sub_3021810(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  _DWORD *v6; // rdx
  __int64 v7; // rcx
  _QWORD *v8; // r12
  _WORD *v9; // rdi
  size_t *v10; // rsi
  size_t v11; // rdx
  void *v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rax
  const char *v15; // rax
  size_t v16; // rdx
  _WORD *v17; // rdi
  unsigned __int8 *v18; // rsi
  unsigned __int64 v19; // rax
  __int64 *v20; // rdi
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rax
  _WORD *v23; // rdx
  unsigned __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  size_t v28; // [rsp+8h] [rbp-148h]
  size_t v29; // [rsp+8h] [rbp-148h]
  __int64 v30; // [rsp+8h] [rbp-148h]
  _QWORD v31[4]; // [rsp+10h] [rbp-140h] BYREF
  __int16 v32; // [rsp+30h] [rbp-120h]
  _QWORD v33[3]; // [rsp+40h] [rbp-110h] BYREF
  __int64 v34; // [rsp+58h] [rbp-F8h]
  _DWORD *v35; // [rsp+60h] [rbp-F0h]
  __int64 v36; // [rsp+68h] [rbp-E8h]
  unsigned __int64 *v37; // [rsp+70h] [rbp-E0h]
  unsigned __int64 v38[3]; // [rsp+80h] [rbp-D0h] BYREF
  _BYTE v39[184]; // [rsp+98h] [rbp-B8h] BYREF

  v36 = 0x100000000LL;
  v37 = v38;
  v38[0] = (unsigned __int64)v39;
  v33[0] = &unk_49DD288;
  v38[1] = 0;
  v38[2] = 128;
  v33[1] = 2;
  v33[2] = 0;
  v34 = 0;
  v35 = 0;
  sub_CB5980((__int64)v33, 0, 0, 0);
  v5 = sub_31DB510(a1, a3);
  v6 = v35;
  v7 = v5;
  if ( (unsigned __int64)(v34 - (_QWORD)v35) <= 6 )
  {
    v30 = v5;
    v27 = sub_CB6200((__int64)v33, ".alias ", 7u);
    v7 = v30;
    v9 = *(_WORD **)(v27 + 32);
    v8 = (_QWORD *)v27;
  }
  else
  {
    *v35 = 1768710446;
    v8 = v33;
    *((_WORD *)v6 + 2) = 29537;
    *((_BYTE *)v6 + 6) = 32;
    v9 = (_WORD *)((char *)v35 + 7);
    v35 = (_DWORD *)((char *)v35 + 7);
  }
  if ( (*(_BYTE *)(v7 + 8) & 1) != 0 )
  {
    v10 = *(size_t **)(v7 - 8);
    v11 = *v10;
    v12 = v10 + 3;
    if ( v8[3] - (_QWORD)v9 >= v11 )
    {
      if ( v11 )
      {
        v29 = v11;
        memcpy(v9, v12, v11);
        v25 = v8[3];
        v9 = (_WORD *)(v8[4] + v29);
        v8[4] = v9;
        if ( (unsigned __int64)(v25 - (_QWORD)v9) > 1 )
          goto LABEL_7;
        goto LABEL_19;
      }
    }
    else
    {
      v13 = sub_CB6200((__int64)v8, (unsigned __int8 *)v12, v11);
      v9 = *(_WORD **)(v13 + 32);
      v8 = (_QWORD *)v13;
    }
  }
  if ( v8[3] - (_QWORD)v9 > 1u )
  {
LABEL_7:
    *v9 = 8236;
    v8[4] += 2LL;
    goto LABEL_8;
  }
LABEL_19:
  v8 = (_QWORD *)sub_CB6200((__int64)v8, (unsigned __int8 *)", ", 2u);
LABEL_8:
  v14 = sub_B325F0(a3);
  v15 = sub_BD5D20(v14);
  v17 = (_WORD *)v8[4];
  v18 = (unsigned __int8 *)v15;
  v19 = v8[3] - (_QWORD)v17;
  if ( v19 < v16 )
  {
    v26 = sub_CB6200((__int64)v8, v18, v16);
    v17 = *(_WORD **)(v26 + 32);
    v8 = (_QWORD *)v26;
    v19 = *(_QWORD *)(v26 + 24) - (_QWORD)v17;
  }
  else if ( v16 )
  {
    v28 = v16;
    memcpy(v17, v18, v16);
    v23 = (_WORD *)(v8[4] + v28);
    v24 = v8[3] - (_QWORD)v23;
    v8[4] = v23;
    v17 = v23;
    if ( v24 > 1 )
      goto LABEL_11;
    goto LABEL_16;
  }
  if ( v19 > 1 )
  {
LABEL_11:
    *v17 = 2619;
    v8[4] += 2LL;
    goto LABEL_12;
  }
LABEL_16:
  sub_CB6200((__int64)v8, (unsigned __int8 *)";\n", 2u);
LABEL_12:
  v20 = *(__int64 **)(a1 + 224);
  v21 = v37[1];
  v22 = *v37;
  v32 = 261;
  v31[0] = v22;
  v31[1] = v21;
  sub_E99A90(v20, (__int64)v31);
  v33[0] = &unk_49DD388;
  sub_CB5840((__int64)v33);
  if ( (_BYTE *)v38[0] != v39 )
    _libc_free(v38[0]);
}
