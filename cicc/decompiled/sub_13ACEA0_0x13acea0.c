// Function: sub_13ACEA0
// Address: 0x13acea0
//
__int64 __fastcall sub_13ACEA0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned int v4; // r14d
  __int64 v5; // rdi
  unsigned __int64 v7[2]; // [rsp+0h] [rbp-50h] BYREF
  _QWORD v8[8]; // [rsp+10h] [rbp-40h] BYREF

  v2 = *(_QWORD *)(a2 + 8LL * *(unsigned __int8 *)(a2 + 280) + 152);
  if ( !v2 )
    return 0;
  v4 = 2;
  while ( *(_DWORD *)(a1 + 40) >= v4 )
  {
    if ( *(_QWORD *)(a2 + 144LL * v4 + 8LL * *(unsigned __int8 *)(a2 + 144LL * v4 + 136) + 8) )
    {
      v5 = *(_QWORD *)(a1 + 8);
      v8[1] = *(_QWORD *)(a2 + 144LL * v4 + 8LL * *(unsigned __int8 *)(a2 + 144LL * v4 + 136) + 8);
      v8[0] = v2;
      v7[0] = (unsigned __int64)v8;
      v7[1] = 0x200000002LL;
      v2 = sub_147DD40(v5, v7, 0, 0);
      if ( (_QWORD *)v7[0] != v8 )
        _libc_free(v7[0]);
      ++v4;
      if ( v2 )
        continue;
    }
    return 0;
  }
  return v2;
}
