// Function: sub_9876C0
// Address: 0x9876c0
//
__int64 *__fastcall sub_9876C0(__int64 *a1)
{
  unsigned int v1; // r14d
  __int64 v2; // r13
  bool v3; // cc
  char v4; // r12
  __int64 *result; // rax
  __int64 v6; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v7; // [rsp+8h] [rbp-38h]
  __int64 v8; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v9; // [rsp+18h] [rbp-28h]

  v7 = *((_DWORD *)a1 + 2);
  if ( v7 > 0x40 )
    sub_C43780(&v6, a1);
  else
    v6 = *a1;
  sub_C46A40(&v6, 1);
  v1 = v7;
  v2 = v6;
  v7 = 0;
  v3 = *((_DWORD *)a1 + 6) <= 0x40u;
  v9 = v1;
  v8 = v6;
  if ( v3 )
    v4 = a1[2] == v6;
  else
    v4 = sub_C43C50(a1 + 2, &v8);
  if ( v1 > 0x40 )
  {
    if ( v2 )
    {
      j_j___libc_free_0_0(v2);
      if ( v7 > 0x40 )
      {
        if ( v6 )
          j_j___libc_free_0_0(v6);
      }
    }
  }
  result = 0;
  if ( v4 )
    return a1;
  return result;
}
