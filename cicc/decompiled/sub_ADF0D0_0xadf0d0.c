// Function: sub_ADF0D0
// Address: 0xadf0d0
//
__int64 __fastcall sub_ADF0D0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rdi
  _QWORD *v20; // rax
  unsigned __int64 v21; // rsi
  __int64 v22; // r12
  __int64 v25; // [rsp+18h] [rbp-118h]
  _QWORD v26[4]; // [rsp+20h] [rbp-110h] BYREF
  _BYTE v27[32]; // [rsp+40h] [rbp-F0h] BYREF
  __int16 v28; // [rsp+60h] [rbp-D0h]
  unsigned int *v29[2]; // [rsp+70h] [rbp-C0h] BYREF
  _BYTE v30[32]; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v31; // [rsp+A0h] [rbp-90h]
  __int64 v32; // [rsp+A8h] [rbp-88h]
  __int16 v33; // [rsp+B0h] [rbp-80h]
  _QWORD *v34; // [rsp+B8h] [rbp-78h]
  void **v35; // [rsp+C0h] [rbp-70h]
  _QWORD *v36; // [rsp+C8h] [rbp-68h]
  __int64 v37; // [rsp+D0h] [rbp-60h]
  int v38; // [rsp+D8h] [rbp-58h]
  __int16 v39; // [rsp+DCh] [rbp-54h]
  char v40; // [rsp+DEh] [rbp-52h]
  __int64 v41; // [rsp+E0h] [rbp-50h]
  __int64 v42; // [rsp+E8h] [rbp-48h]
  void *v43; // [rsp+F0h] [rbp-40h] BYREF
  _QWORD v44[7]; // [rsp+F8h] [rbp-38h] BYREF

  sub_ADDDC0(a1, a4);
  sub_ADDDC0(a1, a5);
  v25 = *(_QWORD *)(a1 + 8);
  v15 = sub_B98A20(a3, a5, v13, v14);
  v16 = sub_B9F6F0(v25, v15);
  v17 = *(_QWORD *)(a1 + 8);
  v26[0] = v16;
  v18 = sub_B9F6F0(v17, a4);
  v19 = *(_QWORD *)(a1 + 8);
  v26[1] = v18;
  v26[2] = sub_B9F6F0(v19, a5);
  v20 = (_QWORD *)(*(_QWORD *)(a6 + 8) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a6 + 8) & 4) != 0 )
    v20 = (_QWORD *)*v20;
  v34 = v20;
  v33 = 0;
  v29[1] = (unsigned int *)0x200000000LL;
  v39 = 512;
  v43 = &unk_49DA100;
  v29[0] = (unsigned int *)v30;
  v35 = &v43;
  v44[0] = &unk_49DA0B0;
  v36 = v44;
  v37 = 0;
  v38 = 0;
  v40 = 7;
  v41 = 0;
  v42 = 0;
  v31 = 0;
  v32 = 0;
  sub_ADD7A0((__int64)v29, a6, a7, a8, SBYTE1(a8));
  v21 = 0;
  v28 = 257;
  if ( a2 )
    v21 = *(_QWORD *)(a2 + 24);
  v22 = sub_921880(v29, v21, a2, (int)v26, 3, (__int64)v27, 0);
  nullsub_61(v44);
  v43 = &unk_49DA100;
  nullsub_63(&v43);
  if ( (_BYTE *)v29[0] != v30 )
    _libc_free(v29[0], v21);
  return v22;
}
