// Function: sub_CBEFB0
// Address: 0xcbefb0
//
__int64 __fastcall sub_CBEFB0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rdi
  __int64 result; // rax

  if ( *(_DWORD *)a1 == 62053 )
  {
    v2 = *(_QWORD *)(a1 + 24);
    if ( v2 )
    {
      if ( *(_DWORD *)v2 == 53829 )
      {
        *(_DWORD *)a1 = 0;
        v3 = *(_QWORD *)(v2 + 8);
        if ( v3 )
          _libc_free(v3, a2);
        v4 = *(_QWORD *)(v2 + 24);
        if ( v4 )
          _libc_free(v4, a2);
        v5 = *(_QWORD *)(v2 + 32);
        if ( v5 )
          _libc_free(v5, a2);
        v6 = *(_QWORD *)(v2 + 96);
        if ( v6 )
          _libc_free(v6, a2);
        return _libc_free(v2, a2);
      }
    }
  }
  return result;
}
