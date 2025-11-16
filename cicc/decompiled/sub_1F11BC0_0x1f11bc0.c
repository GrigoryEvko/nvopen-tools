// Function: sub_1F11BC0
// Address: 0x1f11bc0
//
__int64 __fastcall sub_1F11BC0(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // rbx
  unsigned __int64 v4; // rdi

  result = *(_QWORD *)(a1 + 264);
  if ( result )
  {
    v3 = result + 112LL * *(_QWORD *)(result - 8);
    if ( result != v3 )
    {
      do
      {
        v3 -= 112;
        v4 = *(_QWORD *)(v3 + 24);
        if ( v4 != v3 + 40 )
          _libc_free(v4);
      }
      while ( *(_QWORD *)(a1 + 264) != v3 );
    }
    result = j_j_j___libc_free_0_0(v3 - 8);
  }
  *(_QWORD *)(a1 + 264) = 0;
  *(_DWORD *)(a1 + 472) = 0;
  return result;
}
