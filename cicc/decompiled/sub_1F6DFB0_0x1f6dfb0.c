// Function: sub_1F6DFB0
// Address: 0x1f6dfb0
//
__int64 __fastcall sub_1F6DFB0(unsigned int *a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // r13
  __int64 v4; // rsi
  __int64 *v5; // r12
  unsigned int v6; // ebx
  _QWORD *v7; // r12
  _QWORD *v9; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v10; // [rsp+8h] [rbp-38h]
  _QWORD *v11; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v12; // [rsp+18h] [rbp-28h]

  v3 = (_QWORD *)*a1;
  v4 = *(_QWORD *)(*(_QWORD *)a2 + 88LL);
  v5 = (__int64 *)(*(_QWORD *)(*(_QWORD *)a3 + 88LL) + 24LL);
  v12 = *(_DWORD *)(v4 + 32);
  if ( v12 > 0x40 )
    sub_16A4FD0((__int64)&v11, (const void **)(v4 + 24));
  else
    v11 = *(_QWORD **)(v4 + 24);
  sub_16A7200((__int64)&v11, v5);
  v6 = v12;
  v7 = v11;
  v12 = 0;
  v10 = v6;
  v9 = v11;
  if ( v6 <= 0x40 )
  {
    LOBYTE(v3) = v3 == v11;
    return (unsigned int)v3;
  }
  if ( v6 - (unsigned int)sub_16A57B0((__int64)&v9) <= 0x40 && v3 == (_QWORD *)*v7 )
  {
    LODWORD(v3) = 1;
  }
  else
  {
    LODWORD(v3) = 0;
    if ( !v7 )
      return (unsigned int)v3;
  }
  j_j___libc_free_0_0(v7);
  if ( v12 <= 0x40 || !v11 )
    return (unsigned int)v3;
  j_j___libc_free_0_0(v11);
  return (unsigned int)v3;
}
