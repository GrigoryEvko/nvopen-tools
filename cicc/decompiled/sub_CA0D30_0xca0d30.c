// Function: sub_CA0D30
// Address: 0xca0d30
//
unsigned int __fastcall sub_CA0D30(__int64 a1)
{
  _BYTE *v2; // r12
  __int64 v3; // rsi
  _BYTE *v4; // rdi
  unsigned int result; // eax
  _QWORD v6[4]; // [rsp+0h] [rbp-40h] BYREF
  __int16 v7; // [rsp+20h] [rbp-20h]

  v2 = (_BYTE *)(a1 + 16);
  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(_BYTE **)a1;
  if ( v3 == 1 && *v4 == 45 )
  {
    if ( v4 != v2 )
      return j_j___libc_free_0(v4, *(_QWORD *)(a1 + 16) + 1LL);
  }
  else
  {
    if ( !*(_BYTE *)(a1 + 32) )
    {
      v6[0] = a1;
      v7 = 260;
      sub_C823F0((__int64)v6, 1);
      v4 = *(_BYTE **)a1;
      v3 = *(_QWORD *)(a1 + 8);
    }
    result = sub_C8C850(v4, v3);
    v4 = *(_BYTE **)a1;
    if ( v2 != *(_BYTE **)a1 )
      return j_j___libc_free_0(v4, *(_QWORD *)(a1 + 16) + 1LL);
  }
  return result;
}
