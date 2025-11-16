// Function: sub_16BE410
// Address: 0x16be410
//
__int64 __fastcall sub_16BE410(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _BYTE *v4; // r9
  __int64 v5; // rbx
  _QWORD *v6; // r15
  _QWORD *v7; // r14
  _WORD *v8; // rdi
  unsigned __int64 v9; // rax
  const void *v10; // rbx
  size_t v11; // r13
  _QWORD *v12; // rdx
  __int64 v13; // rax
  unsigned int v14; // r12d
  __int64 v15; // rax
  _QWORD *v18; // [rsp+18h] [rbp-138h]
  _QWORD v19[2]; // [rsp+20h] [rbp-130h] BYREF
  _QWORD v20[2]; // [rsp+30h] [rbp-120h] BYREF
  __int64 v21; // [rsp+40h] [rbp-110h] BYREF
  char v22; // [rsp+50h] [rbp-100h]
  _QWORD v23[2]; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v24; // [rsp+70h] [rbp-E0h]
  _WORD *v25; // [rsp+78h] [rbp-D8h]
  int v26; // [rsp+80h] [rbp-D0h]
  __int64 v27; // [rsp+88h] [rbp-C8h]
  _BYTE *v28; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v29; // [rsp+98h] [rbp-B8h]
  _BYTE v30[176]; // [rsp+A0h] [rbp-B0h] BYREF

  v19[0] = a2;
  v19[1] = a3;
  v23[0] = &unk_49EFBE0;
  v27 = a1;
  v28 = v30;
  v29 = 0x800000000LL;
  v26 = 1;
  v25 = 0;
  v24 = 0;
  v23[1] = 0;
  sub_16D2880(v19, &v28, 124, 0xFFFFFFFFLL, 1);
  v4 = v28;
  v5 = 16LL * (unsigned int)v29;
  v18 = &v28[v5];
  if ( &v28[v5] == v28 )
  {
LABEL_14:
    v14 = 0;
    goto LABEL_17;
  }
  v6 = v28;
  while ( 1 )
  {
    v10 = (const void *)*v6;
    v11 = v6[1];
    sub_16C7620(v20, *v6, v11, 0, 0);
    if ( (v22 & 1) == 0 )
      break;
    v12 = v25;
    if ( (unsigned __int64)(v24 - (_QWORD)v25) > 8 )
    {
      *((_BYTE *)v25 + 8) = 39;
      v7 = v23;
      *v12 = 0x2064656972542020LL;
      v8 = (_WORD *)((char *)v25 + 9);
      v25 = v8;
      v9 = v24 - (_QWORD)v8;
      if ( v11 > v24 - (__int64)v8 )
        goto LABEL_11;
LABEL_4:
      if ( v11 )
      {
        memcpy(v8, v10, v11);
        v15 = v7[2];
        v8 = (_WORD *)(v11 + v7[3]);
        v7[3] = v8;
        v9 = v15 - (_QWORD)v8;
      }
      if ( v9 <= 1 )
        goto LABEL_12;
LABEL_7:
      v6 += 2;
      *v8 = 2599;
      v7[3] += 2LL;
      if ( v18 == v6 )
        goto LABEL_13;
    }
    else
    {
      v7 = (_QWORD *)sub_16E7EE0(v23, "  Tried '", 9);
      v8 = (_WORD *)v7[3];
      v9 = v7[2] - (_QWORD)v8;
      if ( v11 <= v9 )
        goto LABEL_4;
LABEL_11:
      v13 = sub_16E7EE0(v7, (const char *)v10, v11);
      v8 = *(_WORD **)(v13 + 24);
      v7 = (_QWORD *)v13;
      if ( *(_QWORD *)(v13 + 16) - (_QWORD)v8 > 1u )
        goto LABEL_7;
LABEL_12:
      v6 += 2;
      sub_16E7EE0(v7, "'\n", 2);
      if ( v18 == v6 )
      {
LABEL_13:
        v4 = v28;
        goto LABEL_14;
      }
    }
  }
  sub_2240AE0(a4, v20);
  if ( (v22 & 1) == 0 && (__int64 *)v20[0] != &v21 )
    j_j___libc_free_0(v20[0], v21 + 1);
  v4 = v28;
  v14 = 1;
LABEL_17:
  if ( v4 != v30 )
    _libc_free((unsigned __int64)v4);
  sub_16E7BC0(v23);
  return v14;
}
