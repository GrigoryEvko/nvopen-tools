// Function: sub_2E6C600
// Address: 0x2e6c600
//
void __fastcall sub_2E6C600(unsigned __int64 a1)
{
  bool v2; // zf
  __int64 v3; // rbx
  __int64 v4; // rax
  unsigned __int64 v5; // r12
  unsigned __int64 v6; // r14
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi

  v2 = *(_BYTE *)(a1 + 328) == 0;
  *(_QWORD *)a1 = &unk_4A28DC0;
  if ( !v2 )
  {
    v3 = *(_QWORD *)(a1 + 224);
    v4 = *(unsigned int *)(a1 + 232);
    *(_BYTE *)(a1 + 328) = 0;
    v5 = v3 + 8 * v4;
    if ( v3 != v5 )
    {
      do
      {
        v6 = *(_QWORD *)(v5 - 8);
        v5 -= 8LL;
        if ( v6 )
        {
          v7 = *(_QWORD *)(v6 + 24);
          if ( v7 != v6 + 40 )
            _libc_free(v7);
          j_j___libc_free_0(v6);
        }
      }
      while ( v3 != v5 );
      v5 = *(_QWORD *)(a1 + 224);
    }
    if ( v5 != a1 + 240 )
      _libc_free(v5);
    v8 = *(_QWORD *)(a1 + 200);
    if ( v8 != a1 + 216 )
      _libc_free(v8);
  }
  *(_QWORD *)a1 = &unk_49DAF80;
  sub_BB9100(a1);
  j_j___libc_free_0(a1);
}
