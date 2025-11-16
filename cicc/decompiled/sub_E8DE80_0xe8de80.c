// Function: sub_E8DE80
// Address: 0xe8de80
//
__int64 __fastcall sub_E8DE80(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdi
  __int64 v7; // r12
  __int64 *(__fastcall *v8)(_QWORD *, __int64, __int64); // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  char *v19; // [rsp+0h] [rbp-40h] BYREF
  char v20; // [rsp+20h] [rbp-20h]
  char v21; // [rsp+21h] [rbp-1Fh]

  v6 = a1[1];
  v21 = 1;
  v19 = "cfi";
  v20 = 3;
  v7 = sub_E6C380(v6, (__int64 *)&v19, 1, a4, a5);
  v8 = *(__int64 *(__fastcall **)(_QWORD *, __int64, __int64))(*a1 + 208LL);
  if ( v8 == sub_E8DC70 )
  {
    sub_E98820(a1, v7, 0);
    sub_E5CB20(a1[37], v7, v9, v10, v11, v12);
    v13 = sub_E8BB10(a1, 0);
    *(_QWORD *)v7 = v13;
    *(_QWORD *)(v7 + 24) = *(_QWORD *)(v13 + 48);
    *(_BYTE *)(v7 + 9) = *(_BYTE *)(v7 + 9) & 0x8F | 0x10;
    sub_E8DAF0((__int64)a1, v7, v14, v15, v16, v17);
  }
  else
  {
    v8(a1, v7, 0);
  }
  return v7;
}
