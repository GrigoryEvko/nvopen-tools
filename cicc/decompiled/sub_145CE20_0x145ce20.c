// Function: sub_145CE20
// Address: 0x145ce20
//
__int64 __fastcall sub_145CE20(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rbx
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // [rsp+18h] [rbp-C8h] BYREF
  unsigned __int64 v11[2]; // [rsp+20h] [rbp-C0h] BYREF
  _BYTE v12[176]; // [rsp+30h] [rbp-B0h] BYREF

  v11[1] = 0x2000000000LL;
  v11[0] = (unsigned __int64)v12;
  sub_16BD3E0(v11, 0);
  sub_16BD4C0(v11, a2);
  v10 = 0;
  v2 = sub_16BDDE0(a1 + 816, v11, &v10);
  if ( !v2 )
  {
    v4 = sub_16BD760(v11, a1 + 864);
    v6 = v5;
    v7 = v4;
    v8 = sub_145CBF0((__int64 *)(a1 + 864), 40, 16);
    v9 = v10;
    *(_QWORD *)v8 = 0;
    v2 = v8;
    *(_QWORD *)(v8 + 8) = v7;
    *(_QWORD *)(v8 + 16) = v6;
    *(_DWORD *)(v8 + 24) = 0;
    *(_QWORD *)(v8 + 32) = a2;
    sub_16BDA20(a1 + 816, v8, v9);
  }
  if ( (_BYTE *)v11[0] != v12 )
    _libc_free(v11[0]);
  return v2;
}
