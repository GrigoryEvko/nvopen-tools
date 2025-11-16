// Function: sub_22916C0
// Address: 0x22916c0
//
_QWORD *__fastcall sub_22916C0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r13
  unsigned int v4; // r14d
  __int64 *v5; // rdi
  unsigned __int64 v7[2]; // [rsp+0h] [rbp-50h] BYREF
  _QWORD v8[8]; // [rsp+10h] [rbp-40h] BYREF

  v2 = *(_QWORD **)(a2 + 8LL * *(unsigned __int8 *)(a2 + 280) + 216);
  if ( !v2 )
    return 0;
  v4 = 2;
  while ( *(_DWORD *)(a1 + 40) >= v4 )
  {
    if ( *(_QWORD *)(a2 + 144LL * v4 + 8LL * *(unsigned __int8 *)(a2 + 144LL * v4 + 136) + 72) )
    {
      v5 = *(__int64 **)(a1 + 8);
      v8[1] = *(_QWORD *)(a2 + 144LL * v4 + 8LL * *(unsigned __int8 *)(a2 + 144LL * v4 + 136) + 72);
      v8[0] = v2;
      v7[0] = (unsigned __int64)v8;
      v7[1] = 0x200000002LL;
      v2 = sub_DC7EB0(v5, (__int64)v7, 0, 0);
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
