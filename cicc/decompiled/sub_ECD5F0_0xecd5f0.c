// Function: sub_ECD5F0
// Address: 0xecd5f0
//
__int64 __fastcall sub_ECD5F0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // r13
  __int64 result; // rax
  __int64 v6; // r12
  __int64 v7; // rdi

  *(_QWORD *)a1 = &unk_49E49F8;
  v3 = *(_QWORD *)(a1 + 72);
  if ( v3 != a1 + 88 )
  {
    a2 = *(_QWORD *)(a1 + 88) + 1LL;
    j_j___libc_free_0(v3, a2);
  }
  v4 = *(_QWORD *)(a1 + 8);
  result = 5LL * *(unsigned int *)(a1 + 16);
  v6 = v4 + 40LL * *(unsigned int *)(a1 + 16);
  if ( v4 != v6 )
  {
    do
    {
      v6 -= 40;
      if ( *(_DWORD *)(v6 + 32) > 0x40u )
      {
        v7 = *(_QWORD *)(v6 + 24);
        if ( v7 )
          result = j_j___libc_free_0_0(v7);
      }
    }
    while ( v4 != v6 );
    v6 = *(_QWORD *)(a1 + 8);
  }
  if ( v6 != a1 + 24 )
    return _libc_free(v6, a2);
  return result;
}
