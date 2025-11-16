// Function: sub_244FBC0
// Address: 0x244fbc0
//
__int64 __fastcall sub_244FBC0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // rbx
  __int64 v8; // rdi
  bool v9; // al
  __int64 *v10; // rax
  _QWORD *v11; // r15
  __int64 *v12; // rax
  __int64 v13; // rax
  _QWORD *v14; // rax
  __int64 v15; // r9
  __int64 v16; // r10
  int v17; // edx
  _QWORD *v18; // rax
  __int64 v19; // r9
  _QWORD *v20; // rax
  __int64 v21; // r9
  __int64 v22; // r15
  int i; // ebx
  __int64 v24; // r14
  __int64 v26; // [rsp+8h] [rbp-118h]
  _QWORD *v27; // [rsp+8h] [rbp-118h]
  __int64 v28; // [rsp+8h] [rbp-118h]
  __int64 v29; // [rsp+8h] [rbp-118h]
  _QWORD *v30; // [rsp+8h] [rbp-118h]
  __int64 v31; // [rsp+8h] [rbp-118h]
  _QWORD *v32; // [rsp+8h] [rbp-118h]
  _QWORD *v33; // [rsp+10h] [rbp-110h]
  __int64 v35; // [rsp+28h] [rbp-F8h]
  __int64 v36[2]; // [rsp+30h] [rbp-F0h] BYREF
  _QWORD v37[2]; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v38[2]; // [rsp+50h] [rbp-D0h] BYREF
  _QWORD v39[2]; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v40[2]; // [rsp+70h] [rbp-B0h] BYREF
  _QWORD v41[2]; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v42; // [rsp+90h] [rbp-90h] BYREF
  _QWORD *v43; // [rsp+98h] [rbp-88h]
  _QWORD *v44; // [rsp+A0h] [rbp-80h]
  _QWORD *v45; // [rsp+A8h] [rbp-78h]
  _QWORD *v46; // [rsp+B0h] [rbp-70h]
  __int64 v47[2]; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v48; // [rsp+D0h] [rbp-50h] BYREF
  __int16 v49; // [rsp+E0h] [rbp-40h]

  v4 = a3 + 24;
  v5 = *(_QWORD *)a3;
  v6 = *(_QWORD *)(a3 + 32);
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v33 = (_QWORD *)v5;
  if ( v6 == a3 + 24 )
  {
    v7 = 0;
  }
  else
  {
    LODWORD(v7) = 0;
    do
    {
      v8 = v6 - 56;
      if ( !v6 )
        v8 = 0;
      v9 = sub_B2FC80(v8);
      v6 = *(_QWORD *)(v6 + 8);
      LODWORD(v7) = !v9 + (_DWORD)v7;
    }
    while ( v4 != v6 );
    v7 = (int)v7;
  }
  v10 = (__int64 *)sub_BCB2E0(v33);
  v45 = sub_BCD420(v10, 0x20000);
  v11 = (_QWORD *)sub_BCB2D0(v33);
  v12 = (__int64 *)sub_BCB2B0(v33);
  v46 = sub_BCD420(v12, v7);
  v36[0] = (__int64)v37;
  sub_244E240(v36, "_llvm_order_file_buffer", (__int64)"");
  v13 = sub_AD6530((__int64)v45, (__int64)"_llvm_order_file_buffer");
  v47[0] = (__int64)v36;
  v26 = v13;
  v49 = 260;
  BYTE4(v40[0]) = 0;
  v14 = sub_BD2C40(88, unk_3F0FAE8);
  v15 = v26;
  v16 = (__int64)v14;
  if ( v14 )
  {
    v27 = v14;
    sub_B30000((__int64)v14, a3, v45, 0, 3, v15, (__int64)v47, 0, 0, v40[0], 0);
    v16 = (__int64)v27;
  }
  v17 = *(_DWORD *)(a3 + 284);
  v42 = v16;
  v28 = v16;
  sub_ED12E0((__int64)v47, 10, v17, 1u);
  sub_B31A00(v28, v47[0], v47[1]);
  if ( (__int64 *)v47[0] != &v48 )
    j_j___libc_free_0(v47[0]);
  v38[0] = (__int64)v39;
  sub_244E240(v38, "_llvm_order_file_buffer_idx", (__int64)"");
  v29 = sub_AD6530((__int64)v11, (__int64)"_llvm_order_file_buffer_idx");
  v49 = 260;
  v47[0] = (__int64)v38;
  BYTE4(v40[0]) = 0;
  v18 = sub_BD2C40(88, unk_3F0FAE8);
  if ( v18 )
  {
    v19 = v29;
    v30 = v18;
    sub_B30000((__int64)v18, a3, v11, 0, 3, v19, (__int64)v47, 0, 0, v40[0], 0);
    v18 = v30;
  }
  v43 = v18;
  v40[0] = (__int64)v41;
  sub_244E240(v40, "bitmap_0", (__int64)"");
  BYTE4(v35) = 0;
  v31 = sub_AD6530((__int64)v46, (__int64)"bitmap_0");
  v49 = 260;
  v47[0] = (__int64)v40;
  v20 = sub_BD2C40(88, unk_3F0FAE8);
  if ( v20 )
  {
    v21 = v31;
    v32 = v20;
    sub_B30000((__int64)v20, a3, v46, 0, 8, v21, (__int64)v47, 0, 0, v35, 0);
    v20 = v32;
  }
  v44 = v20;
  if ( (_QWORD *)v40[0] != v41 )
    j_j___libc_free_0(v40[0]);
  if ( (_QWORD *)v38[0] != v39 )
    j_j___libc_free_0(v38[0]);
  if ( (_QWORD *)v36[0] != v37 )
    j_j___libc_free_0(v36[0]);
  v22 = *(_QWORD *)(a3 + 32);
  for ( i = 0; v22 != v4; v22 = *(_QWORD *)(v22 + 8) )
  {
    v24 = 0;
    if ( v22 )
      v24 = v22 - 56;
    if ( !sub_B2FC80(v24) )
      sub_244E590(&v42, (__int64 *)a3, v24, i++);
  }
  memset((void *)a1, 0, 0x60u);
  *(_DWORD *)(a1 + 16) = 2;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_BYTE *)(a1 + 28) = 1;
  *(_DWORD *)(a1 + 64) = 2;
  *(_BYTE *)(a1 + 76) = 1;
  return a1;
}
