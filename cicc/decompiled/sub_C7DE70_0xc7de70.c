// Function: sub_C7DE70
// Address: 0xc7de70
//
__int64 __fastcall sub_C7DE70(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // r8
  __int64 v5; // r9
  unsigned __int64 v6; // rax
  int v8; // eax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // [rsp+0h] [rbp-4050h] BYREF
  unsigned __int64 v12; // [rsp+8h] [rbp-4048h] BYREF
  __int64 *v13; // [rsp+10h] [rbp-4040h] BYREF
  unsigned __int64 v14; // [rsp+18h] [rbp-4038h]
  __int64 v15; // [rsp+20h] [rbp-4030h]
  _BYTE v16[16424]; // [rsp+28h] [rbp-4028h] BYREF

  v13 = (__int64 *)v16;
  v14 = 0;
  v15 = 0x4000;
  sub_C85E10(&v11, a2, &v13, 0x4000);
  v6 = v11 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v11 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v11 = 0;
    v12 = v6 | 1;
    v8 = sub_C64300((__int64 *)&v12, a2);
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_DWORD *)a1 = v8;
    v9 = v12;
    *(_QWORD *)(a1 + 8) = v10;
    if ( (v9 & 1) != 0 || (v9 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v12, (__int64)a2);
    if ( (v11 & 1) != 0 || (v11 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v11, (__int64)a2);
  }
  else
  {
    a2 = v13;
    sub_C7DD80(a1, v13, v14, a3, v4, v5);
  }
  if ( v13 != (__int64 *)v16 )
    _libc_free(v13, a2);
  return a1;
}
