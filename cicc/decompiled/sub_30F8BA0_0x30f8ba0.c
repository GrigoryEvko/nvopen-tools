// Function: sub_30F8BA0
// Address: 0x30f8ba0
//
__int64 __fastcall sub_30F8BA0(__int64 a1, __int64 *a2, _QWORD *a3, __int64 a4, __int64 *a5)
{
  __int64 *v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rdi
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned __int64 *v15; // r13
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 *v19; // [rsp+0h] [rbp-50h] BYREF
  __int64 v20; // [rsp+8h] [rbp-48h]
  _QWORD v21[8]; // [rsp+10h] [rbp-40h] BYREF

  v9 = (__int64 *)a3[4];
  v10 = a5[3];
  BYTE4(v20) = 0;
  v11 = a5[4];
  v12 = *a5;
  v13 = *v9;
  v21[2] = v10;
  v21[0] = v12;
  v14 = *(_QWORD *)(v13 + 72);
  v21[1] = v11;
  v21[3] = v14;
  sub_30F7F10((__int64 *)&v19, a3, a5, (__int64)v21, v20);
  if ( v19 )
  {
    sub_30F52A0(*a2, (__int64)v19);
    v15 = v19;
    if ( v19 )
    {
      v16 = v19[18];
      if ( (unsigned __int64 *)v16 != v19 + 20 )
        _libc_free(v16);
      v17 = v15[10];
      if ( (unsigned __int64 *)v17 != v15 + 12 )
        _libc_free(v17);
      if ( (unsigned __int64 *)*v15 != v15 + 2 )
        _libc_free(*v15);
      j_j___libc_free_0((unsigned __int64)v15);
    }
  }
  *(_BYTE *)(a1 + 76) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
