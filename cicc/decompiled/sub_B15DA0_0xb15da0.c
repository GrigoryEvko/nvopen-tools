// Function: sub_B15DA0
// Address: 0xb15da0
//
__int64 __fastcall sub_B15DA0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  void *v5; // rdx
  __int64 v6; // r14
  __int64 v7; // rax
  size_t v8; // rdx
  _BYTE *v9; // rdi
  const void *v10; // rsi
  unsigned __int64 v11; // rax
  _WORD *v12; // rdx
  _BYTE *v13; // rax
  __int64 result; // rax
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  size_t v17; // [rsp+0h] [rbp-C0h]
  _QWORD v18[2]; // [rsp+10h] [rbp-B0h] BYREF
  _QWORD v19[2]; // [rsp+20h] [rbp-A0h] BYREF
  _QWORD v20[2]; // [rsp+30h] [rbp-90h] BYREF
  __int64 v21; // [rsp+40h] [rbp-80h] BYREF
  _QWORD v22[2]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v23; // [rsp+60h] [rbp-60h]
  __int64 v24; // [rsp+68h] [rbp-58h]
  __int64 v25; // [rsp+70h] [rbp-50h]
  __int64 v26; // [rsp+78h] [rbp-48h]
  _QWORD *v27; // [rsp+80h] [rbp-40h]

  v18[0] = v19;
  v26 = 0x100000000LL;
  v18[1] = 0;
  LOBYTE(v19[0]) = 0;
  v22[0] = &unk_49DD210;
  v22[1] = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v27 = v18;
  sub_CB5980(v22, 0, 0, 0);
  sub_B15A70((__int64)v20, a1);
  v4 = sub_CB6200(v22, v20[0], v20[1]);
  v5 = *(void **)(v4 + 32);
  v6 = v4;
  if ( *(_QWORD *)(v4 + 24) - (_QWORD)v5 <= 0xDu )
  {
    v6 = sub_CB6200(v4, ": in function ", 14);
  }
  else
  {
    qmemcpy(v5, ": in function ", 14);
    *(_QWORD *)(v4 + 32) += 14LL;
  }
  v7 = sub_BD5D20(*(_QWORD *)(a1 + 16));
  v9 = *(_BYTE **)(v6 + 32);
  v10 = (const void *)v7;
  v11 = *(_QWORD *)(v6 + 24);
  if ( v11 - (unsigned __int64)v9 < v8 )
  {
    v16 = sub_CB6200(v6, v10, v8);
    v9 = *(_BYTE **)(v16 + 32);
    v6 = v16;
    v11 = *(_QWORD *)(v16 + 24);
  }
  else if ( v8 )
  {
    v17 = v8;
    memcpy(v9, v10, v8);
    v15 = *(_QWORD *)(v6 + 24);
    v9 = (_BYTE *)(*(_QWORD *)(v6 + 32) + v17);
    *(_QWORD *)(v6 + 32) = v9;
    if ( v15 > (unsigned __int64)v9 )
      goto LABEL_6;
    goto LABEL_19;
  }
  if ( v11 > (unsigned __int64)v9 )
  {
LABEL_6:
    *(_QWORD *)(v6 + 32) = v9 + 1;
    *v9 = 32;
    goto LABEL_7;
  }
LABEL_19:
  v6 = sub_CB5D20(v6, 32);
LABEL_7:
  sub_A587F0(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL), v6, 0, 0);
  v12 = *(_WORD **)(v6 + 32);
  if ( *(_QWORD *)(v6 + 24) - (_QWORD)v12 <= 1u )
  {
    v6 = sub_CB6200(v6, ": ", 2);
  }
  else
  {
    *v12 = 8250;
    *(_QWORD *)(v6 + 32) += 2LL;
  }
  sub_CA0E80(a1 + 40, v6);
  v13 = *(_BYTE **)(v6 + 32);
  if ( (unsigned __int64)v13 >= *(_QWORD *)(v6 + 24) )
  {
    sub_CB5D20(v6, 10);
  }
  else
  {
    *(_QWORD *)(v6 + 32) = v13 + 1;
    *v13 = 10;
  }
  if ( (__int64 *)v20[0] != &v21 )
    j_j___libc_free_0(v20[0], v21 + 1);
  if ( v23 != v25 )
    sub_CB5AE0(v22);
  (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a2 + 56LL))(a2, v18);
  v22[0] = &unk_49DD210;
  result = sub_CB5840(v22);
  if ( (_QWORD *)v18[0] != v19 )
    return j_j___libc_free_0(v18[0], v19[0] + 1LL);
  return result;
}
