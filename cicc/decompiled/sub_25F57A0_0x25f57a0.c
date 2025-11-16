// Function: sub_25F57A0
// Address: 0x25f57a0
//
_QWORD *__fastcall sub_25F57A0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // rax
  char v9; // al
  _QWORD *v10; // rsi
  _QWORD *v11; // rdx
  unsigned __int64 v12; // r13
  unsigned __int64 v13; // r15
  __int64 v15; // [rsp+8h] [rbp-B8h] BYREF
  __int64 v16; // [rsp+10h] [rbp-B0h] BYREF
  unsigned __int64 v17; // [rsp+18h] [rbp-A8h] BYREF
  _QWORD v18[2]; // [rsp+20h] [rbp-A0h] BYREF
  __int64 (__fastcall *v19)(_QWORD *, _QWORD *, int); // [rsp+30h] [rbp-90h]
  __int64 (__fastcall *v20)(__int64 *, __int64); // [rsp+38h] [rbp-88h]
  _QWORD v21[2]; // [rsp+40h] [rbp-80h] BYREF
  __int64 (__fastcall *v22)(_QWORD *, _QWORD *, int); // [rsp+50h] [rbp-70h]
  unsigned __int64 (__fastcall *v23)(unsigned __int64 **, __int64); // [rsp+58h] [rbp-68h]
  __int64 v24[12]; // [rsp+60h] [rbp-60h] BYREF

  v7 = *(_QWORD *)(sub_BC0510(a4, &unk_4F82418, a3) + 8);
  v17 = 0;
  v15 = v7;
  v16 = v7;
  v18[0] = v7;
  v20 = sub_25EFA10;
  v19 = sub_25EFAE0;
  v21[0] = &v17;
  v23 = sub_25F01E0;
  v22 = sub_25EFB10;
  v8 = sub_BC0510(a4, &unk_4F87C68, a3);
  v24[4] = (__int64)v18;
  v24[5] = (__int64)v21;
  v24[0] = v8 + 8;
  v24[1] = (__int64)sub_25EF9F0;
  v24[2] = (__int64)&v16;
  v24[3] = (__int64)sub_25EF9D0;
  v24[6] = (__int64)sub_25EFEE0;
  v24[7] = (__int64)&v15;
  v9 = sub_25F56D0(v24, a3);
  v10 = a1 + 4;
  v11 = a1 + 10;
  if ( v9 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v10;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v11;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
  }
  else
  {
    a1[1] = v10;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[7] = v11;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    a1[4] = &qword_4F82400;
    *a1 = 1;
  }
  if ( v22 )
    v22(v21, v21, 3);
  v12 = v17;
  if ( v17 )
  {
    v13 = *(_QWORD *)(v17 + 16);
    if ( v13 )
    {
      sub_FDC110(*(__int64 **)(v17 + 16));
      j_j___libc_free_0(v13);
    }
    j_j___libc_free_0(v12);
  }
  if ( v19 )
    v19(v18, v18, 3);
  return a1;
}
