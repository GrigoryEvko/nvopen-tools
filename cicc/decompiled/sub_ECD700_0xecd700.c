// Function: sub_ECD700
// Address: 0xecd700
//
__int64 __fastcall sub_ECD700(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 result; // rax
  __int64 v5; // r12
  __int64 v6; // rdi

  v3 = *(_QWORD *)(a1 + 16);
  *(_QWORD *)a1 = &unk_49E4A30;
  result = *(unsigned int *)(a1 + 24);
  v5 = v3 + 112 * result;
  if ( v3 != v5 )
  {
    do
    {
      v5 -= 112;
      v6 = *(_QWORD *)(v5 + 8);
      result = v5 + 32;
      if ( v6 != v5 + 32 )
        result = _libc_free(v6, a2);
    }
    while ( v3 != v5 );
    v5 = *(_QWORD *)(a1 + 16);
  }
  if ( v5 != a1 + 32 )
    return _libc_free(v5, a2);
  return result;
}
