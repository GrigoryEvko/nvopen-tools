// Function: sub_FC7680
// Address: 0xfc7680
//
__int64 __fastcall sub_FC7680(__int64 *a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rdi
  __int64 v4; // rbx
  __int64 v5; // r12
  __int64 v6; // r13
  __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 result; // rax

  v2 = *a1;
  if ( *a1 )
  {
    v3 = *(_QWORD *)(v2 + 216);
    if ( v3 != v2 + 232 )
      _libc_free(v3, a2);
    v4 = *(_QWORD *)(v2 + 184);
    v5 = v4 + 16LL * *(unsigned int *)(v2 + 192);
    if ( v4 != v5 )
    {
      do
      {
        v6 = *(_QWORD *)(v5 - 8);
        v5 -= 16;
        if ( v6 )
        {
          sub_AA5290(v6);
          a2 = 80;
          j_j___libc_free_0(v6, 80);
        }
      }
      while ( v4 != v5 );
      v5 = *(_QWORD *)(v2 + 184);
    }
    if ( v5 != v2 + 200 )
      _libc_free(v5, a2);
    v7 = *(_QWORD *)(v2 + 72);
    if ( v7 != v2 + 88 )
      _libc_free(v7, a2);
    v8 = *(_QWORD *)(v2 + 24);
    if ( v8 != v2 + 40 )
      _libc_free(v8, a2);
    return j_j___libc_free_0(v2, 368);
  }
  return result;
}
