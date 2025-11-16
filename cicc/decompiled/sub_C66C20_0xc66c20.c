// Function: sub_C66C20
// Address: 0xc66c20
//
__int64 __fastcall sub_C66C20(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char *v4; // rsi
  _BYTE *v5; // r9
  __int64 v6; // rbx
  char **v7; // r15
  _QWORD *v8; // r13
  _WORD *v9; // rdi
  unsigned __int64 v10; // rax
  char *v11; // rbx
  size_t v12; // r14
  _QWORD *v13; // rdx
  __int64 v14; // rax
  unsigned int v15; // r12d
  __int64 v17; // rax
  char **v19; // [rsp+18h] [rbp-148h]
  _QWORD v20[2]; // [rsp+20h] [rbp-140h] BYREF
  _QWORD v21[2]; // [rsp+30h] [rbp-130h] BYREF
  __int64 v22; // [rsp+40h] [rbp-120h] BYREF
  char v23; // [rsp+50h] [rbp-110h]
  _QWORD v24[3]; // [rsp+60h] [rbp-100h] BYREF
  __int64 v25; // [rsp+78h] [rbp-E8h]
  _WORD *v26; // [rsp+80h] [rbp-E0h]
  __int64 v27; // [rsp+88h] [rbp-D8h]
  __int64 v28; // [rsp+90h] [rbp-D0h]
  _BYTE *v29; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v30; // [rsp+A8h] [rbp-B8h]
  _BYTE v31[176]; // [rsp+B0h] [rbp-B0h] BYREF

  v27 = 0x100000000LL;
  v20[0] = a2;
  v20[1] = a3;
  v24[0] = &unk_49DD210;
  v28 = a1;
  v24[1] = 0;
  v24[2] = 0;
  v25 = 0;
  v26 = 0;
  sub_CB5980(v24, 0, 0, 0);
  v4 = (char *)&v29;
  v29 = v31;
  v30 = 0x800000000LL;
  sub_C93960(v20, &v29, 124, 0xFFFFFFFFLL, 1);
  v5 = v29;
  v6 = 16LL * (unsigned int)v30;
  v19 = (char **)&v29[v6];
  if ( &v29[v6] == v29 )
  {
LABEL_14:
    v15 = 0;
    goto LABEL_15;
  }
  v7 = (char **)v29;
  while ( 1 )
  {
    v11 = *v7;
    v12 = (size_t)v7[1];
    v4 = *v7;
    sub_C86E60(v21, *v7, v12, 0, 0);
    if ( (v23 & 1) == 0 )
      break;
    v13 = v26;
    if ( (unsigned __int64)(v25 - (_QWORD)v26) > 8 )
    {
      *((_BYTE *)v26 + 8) = 39;
      v8 = v24;
      *v13 = 0x2064656972542020LL;
      v9 = (_WORD *)((char *)v26 + 9);
      v26 = v9;
      v10 = v25 - (_QWORD)v9;
      if ( v12 > v25 - (__int64)v9 )
        goto LABEL_11;
LABEL_4:
      if ( v12 )
      {
        v4 = v11;
        memcpy(v9, v11, v12);
        v17 = v8[3];
        v9 = (_WORD *)(v12 + v8[4]);
        v8[4] = v9;
        v10 = v17 - (_QWORD)v9;
      }
      if ( v10 <= 1 )
        goto LABEL_12;
LABEL_7:
      v7 += 2;
      *v9 = 2599;
      v8[4] += 2LL;
      if ( v19 == v7 )
        goto LABEL_13;
    }
    else
    {
      v4 = "  Tried '";
      v8 = (_QWORD *)sub_CB6200(v24, "  Tried '", 9);
      v9 = (_WORD *)v8[4];
      v10 = v8[3] - (_QWORD)v9;
      if ( v12 <= v10 )
        goto LABEL_4;
LABEL_11:
      v4 = v11;
      v14 = sub_CB6200(v8, v11, v12);
      v9 = *(_WORD **)(v14 + 32);
      v8 = (_QWORD *)v14;
      if ( *(_QWORD *)(v14 + 24) - (_QWORD)v9 > 1u )
        goto LABEL_7;
LABEL_12:
      v4 = "'\n";
      v7 += 2;
      sub_CB6200(v8, "'\n", 2);
      if ( v19 == v7 )
      {
LABEL_13:
        v5 = v29;
        goto LABEL_14;
      }
    }
  }
  v4 = (char *)v21;
  sub_2240AE0(a4, v21);
  if ( (v23 & 1) == 0 && (__int64 *)v21[0] != &v22 )
  {
    v4 = (char *)(v22 + 1);
    j_j___libc_free_0(v21[0], v22 + 1);
  }
  v5 = v29;
  v15 = 1;
LABEL_15:
  if ( v5 != v31 )
    _libc_free(v5, v4);
  v24[0] = &unk_49DD210;
  sub_CB5840(v24);
  return v15;
}
