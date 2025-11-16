// Function: sub_35D8900
// Address: 0x35d8900
//
__int64 __fastcall sub_35D8900(_QWORD *a1, unsigned int a2, unsigned int a3)
{
  __int64 v3; // rcx
  __int64 v4; // rsi
  __int64 *v5; // r12
  unsigned int v6; // ebx
  _QWORD *v7; // r12
  __int64 v8; // r13
  unsigned __int64 v10; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v11; // [rsp+8h] [rbp-38h]
  unsigned __int64 v12; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-28h]

  v3 = 5LL * a2;
  v4 = *(_QWORD *)(*a1 + 40LL * a3 + 16);
  v5 = (__int64 *)(*(_QWORD *)(*a1 + 8 * v3 + 8) + 24LL);
  v11 = *(_DWORD *)(v4 + 32);
  if ( v11 > 0x40 )
    sub_C43780((__int64)&v10, (const void **)(v4 + 24));
  else
    v10 = *(_QWORD *)(v4 + 24);
  sub_C46B40((__int64)&v10, v5);
  v6 = v11;
  v7 = (_QWORD *)v10;
  v11 = 0;
  v13 = v6;
  v12 = v10;
  if ( v6 > 0x40 )
  {
    if ( v6 - (unsigned int)sub_C444A0((__int64)&v12) > 0x40 )
    {
      v8 = 0x28F5C28F5C28F5DLL;
    }
    else
    {
      if ( *v7 > 0x28F5C28F5C28F5CuLL )
      {
        v8 = 0x28F5C28F5C28F5DLL;
LABEL_11:
        j_j___libc_free_0_0((unsigned __int64)v7);
        if ( v11 <= 0x40 || !v10 )
          return v8;
        j_j___libc_free_0_0(v10);
        return v8;
      }
      v8 = *v7 + 1LL;
    }
    if ( !v7 )
      return v8;
    goto LABEL_11;
  }
  if ( v10 > 0x28F5C28F5C28F5CLL )
    return 0x28F5C28F5C28F5DLL;
  return v10 + 1;
}
