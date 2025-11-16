// Function: sub_27FE770
// Address: 0x27fe770
//
__int64 __fastcall sub_27FE770(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 *a5)
{
  void *v8; // r15
  __int64 v9; // rsi
  __int16 v10; // ax
  __int64 v11; // rdx
  __int64 v12; // r9
  __int64 v13; // r8
  __int64 v14; // rcx
  __int64 v15; // rbx
  __int64 *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 *v25; // r13
  __int64 *v27; // [rsp+10h] [rbp-100h]
  __int64 **v28; // [rsp+18h] [rbp-F8h]
  __int64 *v29; // [rsp+20h] [rbp-F0h]
  __int64 v30; // [rsp+30h] [rbp-E0h]
  __int64 *v31; // [rsp+38h] [rbp-D8h]
  __int64 v32; // [rsp+40h] [rbp-D0h]
  _QWORD *v33; // [rsp+48h] [rbp-C8h]
  __int64 v34; // [rsp+54h] [rbp-BCh] BYREF
  __int16 v35; // [rsp+5Ch] [rbp-B4h]
  __int64 v36[2]; // [rsp+60h] [rbp-B0h] BYREF
  __int64 *v37; // [rsp+70h] [rbp-A0h]
  _BYTE v38[8]; // [rsp+80h] [rbp-90h] BYREF
  unsigned __int64 v39; // [rsp+88h] [rbp-88h]
  char v40; // [rsp+9Ch] [rbp-74h]
  _BYTE v41[16]; // [rsp+A0h] [rbp-70h] BYREF
  _BYTE v42[8]; // [rsp+B0h] [rbp-60h] BYREF
  unsigned __int64 v43; // [rsp+B8h] [rbp-58h]
  char v44; // [rsp+CCh] [rbp-44h]
  _BYTE v45[64]; // [rsp+D0h] [rbp-40h] BYREF

  if ( !a5[9] )
    sub_C64ED0("LICM requires MemorySSA (loop-mssa)", 0);
  v8 = (void *)(a1 + 32);
  v9 = *(_QWORD *)(**(_QWORD **)(a3 + 32) + 72LL);
  sub_1049690(v36, v9);
  v10 = *((_WORD *)a2 + 4);
  v11 = *a2;
  v12 = a5[1];
  v13 = a5[2];
  v14 = a5[3];
  v15 = a5[9];
  v35 = v10;
  v16 = (__int64 *)a5[4];
  v34 = v11;
  v30 = v12;
  v29 = v16;
  v31 = (__int64 *)v13;
  v28 = (__int64 **)a5[6];
  v32 = v14;
  v33 = (_QWORD *)*a5;
  v27 = (__int64 *)a5[5];
  if ( (unsigned __int8)sub_F6E5B0(a3, v9, *a5, v14, v13, v12)
    || !(unsigned __int8)sub_27FBD50((__int64)&v34, a3, v33, v32, v31, v30, v27, v28, v29, v15, v36, 0) )
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
  sub_22D0390((__int64)v38, a3, v17, v18, v19, v20);
  sub_27ED100((__int64)v38, (__int64)&unk_4F8F810, v21, v22, v23, v24);
  sub_C8CF70(a1, v8, 2, (__int64)v41, (__int64)v38);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v45, (__int64)v42);
  if ( v44 )
  {
    if ( v40 )
      goto LABEL_6;
  }
  else
  {
    _libc_free(v43);
    if ( v40 )
      goto LABEL_6;
  }
  _libc_free(v39);
LABEL_6:
  v25 = v37;
  if ( v37 )
  {
    sub_FDC110(v37);
    j_j___libc_free_0((unsigned __int64)v25);
  }
  return a1;
}
