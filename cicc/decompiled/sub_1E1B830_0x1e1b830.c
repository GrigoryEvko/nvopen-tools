// Function: sub_1E1B830
// Address: 0x1e1b830
//
__int64 __fastcall sub_1E1B830(__int64 a1, int a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rdx
  unsigned int v5; // eax
  __m128i v6; // [rsp+0h] [rbp-40h] BYREF
  __int64 v7; // [rsp+10h] [rbp-30h]
  __int64 v8; // [rsp+18h] [rbp-28h]
  __int64 v9; // [rsp+20h] [rbp-20h]

  if ( a2 > 0 )
  {
    v5 = sub_1E16810(a1, a2, 0, 0, a3);
    if ( v5 == -1 )
      goto LABEL_12;
    result = *(_QWORD *)(a1 + 32) + 40LL * v5;
    if ( !result )
      goto LABEL_12;
  }
  else
  {
    result = *(_QWORD *)(a1 + 32);
    v4 = result + 40LL * *(unsigned int *)(a1 + 40);
    if ( result == v4 )
    {
LABEL_12:
      v6.m128i_i32[2] = a2;
      v6.m128i_i64[0] = 805306368;
      v7 = 0;
      v8 = 0;
      v9 = 0;
      return sub_1E1AFD0(a1, &v6);
    }
    while ( *(_BYTE *)result
         || a2 != *(_DWORD *)(result + 8)
         || (*(_BYTE *)(result + 3) & 0x10) == 0
         || (*(_DWORD *)result & 0xFFF00) != 0 )
    {
      result += 40;
      if ( v4 == result )
        goto LABEL_12;
    }
  }
  return result;
}
