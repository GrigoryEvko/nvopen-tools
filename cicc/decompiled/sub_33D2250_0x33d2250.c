// Function: sub_33D2250
// Address: 0x33d2250
//
__int64 __fastcall sub_33D2250(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // eax
  __int64 v7; // rcx
  unsigned __int64 v8; // rdx
  __int64 result; // rax
  __int64 v10; // [rsp+0h] [rbp-40h]
  unsigned __int64 v11; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v12; // [rsp+18h] [rbp-28h]

  v6 = *(_DWORD *)(a1 + 64);
  v12 = v6;
  if ( v6 > 0x40 )
  {
    sub_C43690((__int64)&v11, -1, 1);
  }
  else
  {
    v7 = -v6;
    v8 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v6;
    if ( !v6 )
      v8 = 0;
    v11 = v8;
  }
  result = sub_33D2050(a1, (__int64)&v11, a2, v7, a5, a6);
  if ( v12 > 0x40 )
  {
    if ( v11 )
    {
      v10 = result;
      j_j___libc_free_0_0(v11);
      return v10;
    }
  }
  return result;
}
