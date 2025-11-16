// Function: sub_2CAFE10
// Address: 0x2cafe10
//
__int64 __fastcall sub_2CAFE10(unsigned __int64 a1, __int64 a2)
{
  int v2; // r8d
  __int64 result; // rax
  unsigned int v4; // edx
  unsigned __int64 v5; // rcx
  unsigned int v6; // [rsp+14h] [rbp-34h]
  unsigned int v7; // [rsp+14h] [rbp-34h]
  unsigned __int64 v8; // [rsp+18h] [rbp-30h] BYREF
  unsigned int v9; // [rsp+20h] [rbp-28h]
  unsigned __int64 v10; // [rsp+28h] [rbp-20h]
  unsigned int v11; // [rsp+30h] [rbp-18h]

  v2 = sub_9AF8B0(a2, a1, 0, 0, 0, 0, 1);
  result = 0;
  if ( v2 )
  {
    sub_9AC3E0((__int64)&v8, a2, a1, 0, 0, 0, 0, 1);
    v4 = v9;
    if ( v9 <= 0x40 )
      v5 = v8;
    else
      v5 = *(_QWORD *)(v8 + 8LL * ((v9 - 1) >> 6));
    result = (unsigned int)((v5 & (1LL << ((unsigned __int8)v9 - 1))) != 0) + 1;
    if ( v11 > 0x40 && v10 )
    {
      v6 = ((v5 & (1LL << ((unsigned __int8)v9 - 1))) != 0) + 1;
      j_j___libc_free_0_0(v10);
      v4 = v9;
      result = v6;
    }
    if ( v4 > 0x40 )
    {
      if ( v8 )
      {
        v7 = result;
        j_j___libc_free_0_0(v8);
        return v7;
      }
    }
  }
  return result;
}
