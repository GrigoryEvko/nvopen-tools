// Function: sub_1B75110
// Address: 0x1b75110
//
__int64 __fastcall sub_1B75110(__int64 *a1)
{
  __int64 v1; // r14
  unsigned __int64 v2; // rdi
  __int64 v3; // rbx
  unsigned __int64 v4; // r12
  __int64 v5; // r13
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  __int64 result; // rax

  v1 = *a1;
  if ( *a1 )
  {
    v2 = *(_QWORD *)(v1 + 216);
    if ( v2 != v1 + 232 )
      _libc_free(v2);
    v3 = *(_QWORD *)(v1 + 184);
    v4 = v3 + 16LL * *(unsigned int *)(v1 + 192);
    if ( v3 != v4 )
    {
      do
      {
        v5 = *(_QWORD *)(v4 - 8);
        v4 -= 16LL;
        if ( v5 )
        {
          sub_157EF40(v5);
          j_j___libc_free_0(v5, 64);
        }
      }
      while ( v3 != v4 );
      v4 = *(_QWORD *)(v1 + 184);
    }
    if ( v4 != v1 + 200 )
      _libc_free(v4);
    v6 = *(_QWORD *)(v1 + 72);
    if ( v6 != v1 + 88 )
      _libc_free(v6);
    v7 = *(_QWORD *)(v1 + 24);
    if ( v7 != v1 + 40 )
      _libc_free(v7);
    return j_j___libc_free_0(v1, 360);
  }
  return result;
}
