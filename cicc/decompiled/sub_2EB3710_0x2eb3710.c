// Function: sub_2EB3710
// Address: 0x2eb3710
//
__int64 __fastcall sub_2EB3710(__int64 a1)
{
  bool v2; // zf
  __int64 v4; // rbx
  __int64 v5; // rax
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // r13
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi

  v2 = *(_BYTE *)(a1 + 352) == 0;
  *(_QWORD *)a1 = &unk_4A298B8;
  if ( !v2 )
  {
    v4 = *(_QWORD *)(a1 + 248);
    v5 = *(unsigned int *)(a1 + 256);
    *(_BYTE *)(a1 + 352) = 0;
    v6 = v4 + 8 * v5;
    if ( v4 != v6 )
    {
      do
      {
        v7 = *(_QWORD *)(v6 - 8);
        v6 -= 8LL;
        if ( v7 )
        {
          v8 = *(_QWORD *)(v7 + 24);
          if ( v8 != v7 + 40 )
            _libc_free(v8);
          j_j___libc_free_0(v7);
        }
      }
      while ( v4 != v6 );
      v6 = *(_QWORD *)(a1 + 248);
    }
    if ( v6 != a1 + 264 )
      _libc_free(v6);
    v9 = *(_QWORD *)(a1 + 200);
    if ( v9 != a1 + 216 )
      _libc_free(v9);
  }
  *(_QWORD *)a1 = &unk_49DAF80;
  return sub_BB9100(a1);
}
