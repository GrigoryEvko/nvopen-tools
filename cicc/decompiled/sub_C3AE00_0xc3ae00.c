// Function: sub_C3AE00
// Address: 0xc3ae00
//
__int64 __fastcall sub_C3AE00(__int64 a1, __int64 *a2)
{
  __int64 *v2; // rax
  __int64 v3; // rbx
  __int64 result; // rax
  bool v5; // [rsp+Fh] [rbp-51h] BYREF
  __int64 v6; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v7; // [rsp+18h] [rbp-48h]
  __int64 *v8; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v9; // [rsp+28h] [rbp-38h]

  if ( *((_DWORD *)a2 + 2) > 0x40u )
    a2 = (__int64 *)*a2;
  v2 = (__int64 *)*a2;
  v3 = a2[1];
  v9 = 64;
  v8 = v2;
  sub_C37CF0(a1, (unsigned __int64)&v8);
  if ( v9 > 0x40 && v8 )
    j_j___libc_free_0_0(v8);
  sub_C396A0(a1, dword_3F65580, 1, &v5);
  result = *(unsigned __int8 *)(a1 + 20);
  if ( (result & 6) != 0 )
  {
    result &= 7u;
    if ( (_BYTE)result != 3 )
    {
      v6 = v3;
      v7 = 64;
      sub_C3B160(&v8, dword_3F657A0, &v6);
      if ( v7 > 0x40 )
      {
        if ( v6 )
          j_j___libc_free_0_0(v6);
      }
      sub_C396A0((__int64)&v8, dword_3F65580, 1, &v5);
      sub_C3ADF0(a1, (__int64)&v8, 1);
      return sub_C338F0((__int64)&v8);
    }
  }
  return result;
}
