// Function: sub_2B11550
// Address: 0x2b11550
//
__int64 __fastcall sub_2B11550(_QWORD *a1, unsigned int a2)
{
  unsigned __int8 *v2; // r12
  int v3; // eax
  __int64 v5; // r12
  _BYTE v6[24]; // [rsp+0h] [rbp-C0h] BYREF
  char *v7; // [rsp+18h] [rbp-A8h]
  char v8; // [rsp+28h] [rbp-98h] BYREF
  char *v9; // [rsp+48h] [rbp-78h]
  char v10; // [rsp+58h] [rbp-68h] BYREF

  v2 = *(unsigned __int8 **)(*(_QWORD *)(*a1 + 32LL) + 8LL * a2);
  v3 = sub_9B78C0((__int64)v2, *(__int64 **)(a1[1] + 3304LL));
  if ( !v3 )
    return sub_DFD7B0(*(_QWORD *)(a1[1] + 3296LL));
  sub_DF86E0((__int64)v6, v3, v2, 1, 0, 0, 0);
  v5 = sub_DFD690(*(_QWORD *)(a1[1] + 3296LL), (__int64)v6);
  if ( v9 != &v10 )
    _libc_free((unsigned __int64)v9);
  if ( v7 != &v8 )
    _libc_free((unsigned __int64)v7);
  return v5;
}
