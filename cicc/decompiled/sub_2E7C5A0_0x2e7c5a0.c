// Function: sub_2E7C5A0
// Address: 0x2e7c5a0
//
__int64 __fastcall sub_2E7C5A0(unsigned __int64 *a1, const void **a2, __int64 a3)
{
  unsigned __int64 v4; // rsi
  unsigned __int64 v6; // rdi
  __int64 v7[3]; // [rsp+0h] [rbp-30h] BYREF
  int v8; // [rsp+18h] [rbp-18h]

  sub_2E79A00((__int64)v7, a2, a3);
  v4 = a1[2];
  if ( v4 == a1[3] )
  {
    sub_2E7C370(a1 + 1, (char *)v4, v7);
    v6 = v7[0];
  }
  else
  {
    if ( v4 )
    {
      *(_QWORD *)v4 = v7[0];
      *(_QWORD *)(v4 + 8) = v7[1];
      *(_QWORD *)(v4 + 16) = v7[2];
      *(_DWORD *)(v4 + 24) = v8;
      a1[2] += 32LL;
      return (unsigned int)((__int64)(a1[2] - a1[1]) >> 5) - 1;
    }
    v6 = v7[0];
    a1[2] = 32;
  }
  if ( v6 )
    j_j___libc_free_0(v6);
  return (unsigned int)((__int64)(a1[2] - a1[1]) >> 5) - 1;
}
