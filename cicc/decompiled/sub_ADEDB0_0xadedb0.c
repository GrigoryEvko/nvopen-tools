// Function: sub_ADEDB0
// Address: 0xadedb0
//
unsigned __int64 __fastcall sub_ADEDB0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v8; // r11
  __int64 v12; // rdi
  __int64 v13; // r10
  __int16 v14; // r9
  __int64 v15; // rbx
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rdi
  _QWORD *v23; // rax
  __int64 v24; // rdx
  unsigned __int64 v25; // rsi
  unsigned __int64 v26; // r12
  __int64 v28; // rax
  __int64 v29; // rbx
  __int64 v30; // rax
  __int16 v31; // [rsp+8h] [rbp-128h]
  __int64 v32; // [rsp+10h] [rbp-120h]
  __int64 v33; // [rsp+18h] [rbp-118h]
  _QWORD v35[4]; // [rsp+20h] [rbp-110h] BYREF
  char v36[32]; // [rsp+40h] [rbp-F0h] BYREF
  __int16 v37; // [rsp+60h] [rbp-D0h]
  unsigned int *v38[2]; // [rsp+70h] [rbp-C0h] BYREF
  _BYTE v39[32]; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v40; // [rsp+A0h] [rbp-90h]
  __int64 v41; // [rsp+A8h] [rbp-88h]
  __int16 v42; // [rsp+B0h] [rbp-80h]
  _QWORD *v43; // [rsp+B8h] [rbp-78h]
  void **v44; // [rsp+C0h] [rbp-70h]
  _QWORD *v45; // [rsp+C8h] [rbp-68h]
  __int64 v46; // [rsp+D0h] [rbp-60h]
  int v47; // [rsp+D8h] [rbp-58h]
  __int16 v48; // [rsp+DCh] [rbp-54h]
  char v49; // [rsp+DEh] [rbp-52h]
  __int64 v50; // [rsp+E0h] [rbp-50h]
  __int64 v51; // [rsp+E8h] [rbp-48h]
  void *v52; // [rsp+F0h] [rbp-40h] BYREF
  _QWORD v53[7]; // [rsp+F8h] [rbp-38h] BYREF

  v8 = a5;
  v12 = *a1;
  v13 = a7;
  v14 = a8;
  if ( *(_BYTE *)(v12 + 872) )
  {
    v28 = sub_B12860(a2, a3, a4, a5);
    v29 = v28 | 4;
    sub_ADE610((__int64)a1, v28, a7, a8);
    return v29;
  }
  else
  {
    if ( !a1[3] )
    {
      v30 = sub_B6E160(v12, 69, 0, 0);
      v14 = a8;
      v13 = a7;
      a1[3] = v30;
      v8 = a5;
    }
    v31 = v14;
    v32 = v13;
    v33 = v8;
    sub_ADDDC0((__int64)a1, a3);
    sub_ADDDC0((__int64)a1, a4);
    v15 = a1[1];
    v18 = sub_B98A20(a2, a4, v16, v17);
    v19 = sub_B9F6F0(v15, v18);
    v20 = a1[1];
    v35[0] = v19;
    v21 = sub_B9F6F0(v20, a3);
    v22 = a1[1];
    v35[1] = v21;
    v35[2] = sub_B9F6F0(v22, a4);
    v23 = (_QWORD *)(*(_QWORD *)(v33 + 8) & 0xFFFFFFFFFFFFFFF8LL);
    if ( (*(_QWORD *)(v33 + 8) & 4) != 0 )
      v23 = (_QWORD *)*v23;
    v43 = v23;
    v42 = 0;
    v44 = &v52;
    v52 = &unk_49DA100;
    v38[1] = (unsigned int *)0x200000000LL;
    v48 = 512;
    v53[0] = &unk_49DA0B0;
    v38[0] = (unsigned int *)v39;
    v45 = v53;
    v46 = 0;
    v47 = 0;
    v49 = 7;
    v50 = 0;
    v51 = 0;
    v40 = 0;
    v41 = 0;
    sub_ADD7A0((__int64)v38, v33, v32, v31, SHIBYTE(v31));
    v24 = a1[3];
    v25 = 0;
    v37 = 257;
    if ( v24 )
      v25 = *(_QWORD *)(v24 + 24);
    v26 = sub_921880(v38, v25, v24, (int)v35, 3, (__int64)v36, 0) & 0xFFFFFFFFFFFFFFFBLL;
    nullsub_61(v53);
    v52 = &unk_49DA100;
    nullsub_63(&v52);
    if ( (_BYTE *)v38[0] != v39 )
      _libc_free(v38[0], v25);
  }
  return v26;
}
