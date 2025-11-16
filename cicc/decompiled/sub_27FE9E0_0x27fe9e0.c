// Function: sub_27FE9E0
// Address: 0x27fe9e0
//
__int64 __fastcall sub_27FE9E0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 *a5)
{
  void *v8; // r15
  __int64 v9; // rsi
  char v10; // al
  __int64 v11; // rdx
  __int64 v12; // r9
  __int64 v13; // r8
  __int64 *v14; // rax
  __int64 v15; // rcx
  __int64 v16; // r13
  __int64 v17; // rbx
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 *v34; // r13
  __int64 *v36; // [rsp+10h] [rbp-100h]
  __int64 **v37; // [rsp+18h] [rbp-F8h]
  __int64 *v38; // [rsp+20h] [rbp-F0h]
  __int64 v39; // [rsp+30h] [rbp-E0h]
  __int64 *v40; // [rsp+38h] [rbp-D8h]
  __int64 v41; // [rsp+40h] [rbp-D0h]
  _QWORD *v42; // [rsp+48h] [rbp-C8h]
  __int64 v43; // [rsp+54h] [rbp-BCh] BYREF
  char v44; // [rsp+5Ch] [rbp-B4h]
  char v45; // [rsp+5Dh] [rbp-B3h]
  __int64 v46[2]; // [rsp+60h] [rbp-B0h] BYREF
  __int64 *v47; // [rsp+70h] [rbp-A0h]
  _BYTE v48[8]; // [rsp+80h] [rbp-90h] BYREF
  unsigned __int64 v49; // [rsp+88h] [rbp-88h]
  char v50; // [rsp+9Ch] [rbp-74h]
  _BYTE v51[16]; // [rsp+A0h] [rbp-70h] BYREF
  _BYTE v52[8]; // [rsp+B0h] [rbp-60h] BYREF
  unsigned __int64 v53; // [rsp+B8h] [rbp-58h]
  char v54; // [rsp+CCh] [rbp-44h]
  _BYTE v55[64]; // [rsp+D0h] [rbp-40h] BYREF

  if ( !a5[9] )
    sub_C64ED0("LNICM requires MemorySSA (loop-mssa)", 0);
  v8 = (void *)(a1 + 32);
  v9 = *(_QWORD *)(**(_QWORD **)(**(_QWORD **)(a3 + 8) + 32LL) + 72LL);
  sub_1049690(v46, v9);
  v10 = *((_BYTE *)a2 + 8);
  v11 = *a2;
  v45 = 0;
  v12 = a5[1];
  v13 = a5[2];
  v44 = v10;
  v14 = *(__int64 **)(a3 + 8);
  v15 = a5[3];
  v43 = v11;
  v16 = *v14;
  v39 = v12;
  v17 = a5[9];
  v40 = (__int64 *)v13;
  v38 = (__int64 *)a5[4];
  v41 = v15;
  v37 = (__int64 **)a5[6];
  v42 = (_QWORD *)*a5;
  v36 = (__int64 *)a5[5];
  if ( (unsigned __int8)sub_F6E5B0(*v14, v9, *a5, v15, v13, v12)
    || !(unsigned __int8)sub_27FBD50((__int64)&v43, v16, v42, v41, v40, v39, v36, v37, v38, v17, v46, 1) )
  {
    *(_QWORD *)(a1 + 8) = v8;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    goto LABEL_6;
  }
  sub_22D0390((__int64)v48, v16, v18, v19, v20, v21);
  sub_27ED100((__int64)v48, (__int64)&unk_4F81450, v22, v23, v24, v25);
  sub_27ED100((__int64)v48, (__int64)&unk_4F875F0, v26, v27, v28, v29);
  sub_27ED100((__int64)v48, (__int64)&unk_4F8F810, v30, v31, v32, v33);
  sub_C8CF70(a1, v8, 2, (__int64)v51, (__int64)v48);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v55, (__int64)v52);
  if ( v54 )
  {
    if ( v50 )
      goto LABEL_6;
  }
  else
  {
    _libc_free(v53);
    if ( v50 )
      goto LABEL_6;
  }
  _libc_free(v49);
LABEL_6:
  v34 = v47;
  if ( v47 )
  {
    sub_FDC110(v47);
    j_j___libc_free_0((unsigned __int64)v34);
  }
  return a1;
}
