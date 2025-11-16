// Function: sub_1BFA590
// Address: 0x1bfa590
//
__int64 __fastcall sub_1BFA590(unsigned __int64 a1)
{
  unsigned int v1; // r12d
  unsigned __int8 v3; // [rsp+Fh] [rbp-31h] BYREF
  __int64 v4; // [rsp+10h] [rbp-30h] BYREF
  __int64 v5; // [rsp+18h] [rbp-28h]
  __int64 v6; // [rsp+20h] [rbp-20h]
  __int64 v7; // [rsp+28h] [rbp-18h]

  v1 = 1;
  if ( *(_BYTE *)(a1 + 16) > 0x17u )
  {
    v4 = 0;
    v5 = 0;
    v6 = 0;
    v7 = 0;
    v3 = 1;
    sub_1BFA200(a1, (__int64)&v4, &v3);
    v1 = v3;
    j___libc_free_0(v5);
  }
  return v1;
}
