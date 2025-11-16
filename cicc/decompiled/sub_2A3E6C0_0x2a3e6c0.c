// Function: sub_2A3E6C0
// Address: 0x2a3e6c0
//
void __fastcall sub_2A3E6C0(char *a1, __int64 a2, __int64 a3)
{
  unsigned int *v4; // [rsp+0h] [rbp-60h] BYREF
  __int64 v5; // [rsp+8h] [rbp-58h]
  _BYTE v6[80]; // [rsp+10h] [rbp-50h] BYREF

  v4 = (unsigned int *)v6;
  v5 = 0xC00000000LL;
  if ( (unsigned __int8)sub_BC8C10((__int64)a1, (__int64)&v4) )
    sub_2A3E480(a1, v4, (unsigned int)v5, a2, a3);
  if ( v4 != (unsigned int *)v6 )
    _libc_free((unsigned __int64)v4);
}
