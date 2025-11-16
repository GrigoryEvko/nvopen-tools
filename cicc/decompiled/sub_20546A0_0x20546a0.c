// Function: sub_20546A0
// Address: 0x20546a0
//
__int64 __fastcall sub_20546A0(__int64 a1, _QWORD *a2, unsigned int a3, unsigned int a4)
{
  __int64 v4; // r12
  __int64 v5; // rsi
  __int64 *v6; // r12
  unsigned int v7; // ebx
  _QWORD *v8; // r12
  __int64 v9; // r13
  unsigned __int64 v11; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v12; // [rsp+8h] [rbp-38h]
  unsigned __int64 v13; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-28h]

  v4 = *(_QWORD *)(*a2 + 40LL * a3 + 8);
  v5 = *(_QWORD *)(*a2 + 40LL * a4 + 16);
  v6 = (__int64 *)(v4 + 24);
  v12 = *(_DWORD *)(v5 + 32);
  if ( v12 > 0x40 )
    sub_16A4FD0((__int64)&v11, (const void **)(v5 + 24));
  else
    v11 = *(_QWORD *)(v5 + 24);
  sub_16A7590((__int64)&v11, v6);
  v7 = v12;
  v8 = (_QWORD *)v11;
  v12 = 0;
  v14 = v7;
  v13 = v11;
  if ( v7 > 0x40 )
  {
    if ( v7 - (unsigned int)sub_16A57B0((__int64)&v13) > 0x40 )
    {
      v9 = 0x28F5C28F5C28F5DLL;
    }
    else
    {
      if ( *v8 > 0x28F5C28F5C28F5CuLL )
      {
        v9 = 0x28F5C28F5C28F5DLL;
LABEL_11:
        j_j___libc_free_0_0(v8);
        if ( v12 <= 0x40 || !v11 )
          return v9;
        j_j___libc_free_0_0(v11);
        return v9;
      }
      v9 = *v8 + 1LL;
    }
    if ( !v8 )
      return v9;
    goto LABEL_11;
  }
  if ( v11 > 0x28F5C28F5C28F5CLL )
    return 0x28F5C28F5C28F5DLL;
  return v11 + 1;
}
