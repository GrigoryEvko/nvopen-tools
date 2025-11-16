// Function: sub_25AB5A0
// Address: 0x25ab5a0
//
__int64 __fastcall sub_25AB5A0(__int64 a1)
{
  _DWORD *v1; // rdx
  _DWORD *v2; // rax
  unsigned int v3; // r12d
  _BYTE *v5; // [rsp+0h] [rbp-60h] BYREF
  __int64 v6; // [rsp+8h] [rbp-58h]
  _BYTE v7[80]; // [rsp+10h] [rbp-50h] BYREF

  v5 = v7;
  v6 = 0x400000000LL;
  sub_B9A9D0(a1, (__int64)&v5);
  v1 = &v5[16 * (unsigned int)v6];
  if ( v1 == (_DWORD *)v5 )
  {
LABEL_6:
    v3 = 0;
  }
  else
  {
    v2 = v5;
    while ( !*v2 )
    {
      v2 += 4;
      if ( v1 == v2 )
        goto LABEL_6;
    }
    v3 = 1;
  }
  if ( v5 != v7 )
    _libc_free((unsigned __int64)v5);
  return v3;
}
