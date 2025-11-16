// Function: sub_169CEC0
// Address: 0x169cec0
//
__int64 __fastcall sub_169CEC0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 result; // rax
  bool v5; // [rsp+Fh] [rbp-51h] BYREF
  __int64 v6; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v7; // [rsp+18h] [rbp-48h]
  __int64 v8; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v9; // [rsp+28h] [rbp-38h]

  if ( *(_DWORD *)(a2 + 8) > 0x40u )
    a2 = *(_QWORD *)a2;
  v2 = *(_QWORD *)a2;
  v3 = *(_QWORD *)(a2 + 8);
  v9 = 64;
  v8 = v2;
  sub_169AFE0(a1, &v8);
  if ( v9 > 0x40 && v8 )
    j_j___libc_free_0_0(v8);
  sub_16995F0(a1, word_42AE980, 0, &v5);
  result = *(unsigned __int8 *)(a1 + 18);
  if ( (result & 6) != 0 )
  {
    result &= 7u;
    if ( (_BYTE)result != 3 )
    {
      v6 = v3;
      v7 = 64;
      sub_169D050(&v8, &unk_42AE9D0, &v6);
      if ( v7 > 0x40 )
      {
        if ( v6 )
          j_j___libc_free_0_0(v6);
      }
      sub_16995F0((__int64)&v8, word_42AE980, 0, &v5);
      sub_169CEB0((__int16 **)a1, &v8, 0);
      return sub_1698460((__int64)&v8);
    }
  }
  return result;
}
