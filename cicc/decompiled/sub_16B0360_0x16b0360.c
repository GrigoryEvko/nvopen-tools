// Function: sub_16B0360
// Address: 0x16b0360
//
__int64 __fastcall sub_16B0360(__int64 a1)
{
  unsigned __int64 v2; // r8
  __int64 v3; // r13
  __int64 v4; // r13
  __int64 v5; // rbx
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  __int64 result; // rax

  if ( a1 )
  {
    v2 = *(_QWORD *)(a1 + 128);
    if ( *(_DWORD *)(a1 + 140) )
    {
      v3 = *(unsigned int *)(a1 + 136);
      if ( (_DWORD)v3 )
      {
        v4 = 8 * v3;
        v5 = 0;
        do
        {
          v6 = *(_QWORD *)(v2 + v5);
          if ( v6 != -8 && v6 )
          {
            _libc_free(v6);
            v2 = *(_QWORD *)(a1 + 128);
          }
          v5 += 8;
        }
        while ( v4 != v5 );
      }
    }
    _libc_free(v2);
    v7 = *(_QWORD *)(a1 + 80);
    if ( v7 != a1 + 96 )
      _libc_free(v7);
    v8 = *(_QWORD *)(a1 + 32);
    if ( v8 != a1 + 48 )
      _libc_free(v8);
    return j_j___libc_free_0(a1, 168);
  }
  return result;
}
