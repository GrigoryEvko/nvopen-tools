// Function: sub_257AB00
// Address: 0x257ab00
//
__int64 __fastcall sub_257AB00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  __int64 v8; // rcx
  unsigned __int64 v9; // rax
  __int64 v10; // r13
  unsigned __int64 v11; // rsi
  __int64 v12; // rdi
  __int64 (__fastcall *v13)(__int64); // rax
  _BYTE *v14; // rsi
  unsigned int v15; // r13d
  unsigned int v16; // eax
  int *v17; // rbx
  unsigned __int64 v19[2]; // [rsp+0h] [rbp-50h] BYREF
  __int64 v20; // [rsp+10h] [rbp-40h]
  unsigned __int64 v21[2]; // [rsp+18h] [rbp-38h] BYREF
  _BYTE v22[40]; // [rsp+28h] [rbp-28h] BYREF

  v7 = *(_QWORD *)a2;
  v8 = *(unsigned int *)(a2 + 16);
  v21[0] = (unsigned __int64)v22;
  v21[1] = 0;
  v20 = v7;
  if ( (_DWORD)v8 )
  {
    sub_2538240((__int64)v21, (char **)(a2 + 8), a3, v8, a5, a6);
    v7 = v20;
  }
  v9 = sub_B43CB0(v7);
  v10 = *(_QWORD *)(a1 + 8);
  sub_250D230(v19, v9, 4, 0);
  v11 = v19[0];
  v12 = sub_257A720(v10, v19[0], v19[1], *(_QWORD *)(a1 + 16), 0, 0, 1);
  if ( v12 )
  {
    v13 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v12 + 48LL);
    if ( v13 == sub_2534FA0 )
      v14 = (_BYTE *)(v12 + 88);
    else
      v14 = (_BYTE *)((__int64 (__fastcall *)(__int64, unsigned __int64))v13)(v12, v11);
    v15 = 1;
    v16 = sub_255B4C0((_BYTE *)(*(_QWORD *)(a1 + 16) + 88LL), v14);
    v17 = *(int **)a1;
    *v17 = sub_250C0B0(*v17, v16);
  }
  else
  {
    v15 = 0;
  }
  if ( (_BYTE *)v21[0] != v22 )
    _libc_free(v21[0]);
  return v15;
}
