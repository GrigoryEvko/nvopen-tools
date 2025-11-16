// Function: sub_11CA4B0
// Address: 0x11ca4b0
//
__int64 __fastcall sub_11CA4B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 *a7)
{
  __int64 *v9; // r13
  bool v10; // al
  __int64 v11; // r8
  unsigned __int64 v12; // rax
  __int64 *v13; // rdi
  __int64 v14; // rax
  unsigned int v15; // eax
  unsigned __int64 v16; // rax
  __int64 v17; // r15
  unsigned __int8 *v18; // rdx
  unsigned __int8 *v19; // r13
  unsigned __int8 *v20; // rax
  __int64 *v22; // [rsp+8h] [rbp-B8h]
  unsigned __int64 v23; // [rsp+10h] [rbp-B0h]
  __int64 v24; // [rsp+18h] [rbp-A8h]
  __int64 v27; // [rsp+28h] [rbp-98h]
  unsigned __int64 v28; // [rsp+38h] [rbp-88h] BYREF
  _QWORD v29[4]; // [rsp+40h] [rbp-80h] BYREF
  _QWORD v30[2]; // [rsp+60h] [rbp-60h] BYREF
  _QWORD v31[2]; // [rsp+70h] [rbp-50h] BYREF
  __int64 v32; // [rsp+80h] [rbp-40h]
  __int64 v33; // [rsp+88h] [rbp-38h]

  v9 = (__int64 *)sub_AA4B30(*(_QWORD *)(a5 + 48));
  v10 = sub_11C99B0(v9, a7, 0x79u);
  v11 = 0;
  if ( v10 )
  {
    v28 = 0;
    LODWORD(v30[0]) = 41;
    v12 = sub_A79F10((__int64 *)*v9, 0xFFFFFFFF, (int *)v30, 1);
    v13 = *(__int64 **)(a5 + 72);
    v28 = v12;
    v22 = (__int64 *)sub_BCE3C0(v13, 0);
    v14 = sub_AA4B30(*(_QWORD *)(a5 + 48));
    v15 = sub_97FA80(*a7, v14);
    v24 = sub_BCD140(*(_QWORD **)(a5 + 72), v15);
    v23 = sub_A7B050((__int64 *)*v9, &v28, 1);
    v32 = v24;
    v33 = v24;
    v30[0] = v31;
    v31[0] = v22;
    v31[1] = v22;
    v30[1] = 0x400000004LL;
    v16 = sub_BCF480(v22, v31, 4, 0);
    v17 = sub_11C96C0((__int64)v9, a7, 0x79u, v16, v23);
    v19 = v18;
    if ( (_QWORD *)v30[0] != v31 )
      _libc_free(v30[0], a7);
    LOWORD(v32) = 257;
    v29[0] = a1;
    v29[2] = a3;
    v29[1] = a2;
    v29[3] = a4;
    v27 = sub_921880((unsigned int **)a5, v17, (int)v19, (int)v29, 4, (__int64)v30, 0);
    v20 = sub_BD3990(v19, v17);
    v11 = v27;
    if ( !*v20 )
      *(_WORD *)(v27 + 2) = *(_WORD *)(v27 + 2) & 0xF003 | (4 * ((*((_WORD *)v20 + 1) >> 4) & 0x3FF));
  }
  return v11;
}
