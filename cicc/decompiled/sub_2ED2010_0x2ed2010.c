// Function: sub_2ED2010
// Address: 0x2ed2010
//
void __fastcall sub_2ED2010(unsigned __int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // r14
  __int64 v4; // rbx
  __int64 v5; // r15
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi

  *(_QWORD *)a1 = off_4A2A1F0;
  v2 = *(unsigned int *)(a1 + 384);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD *)(a1 + 368);
    v4 = v3 + 88 * v2;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v3 <= 0xFFFFFFFD )
        {
          v5 = *(_QWORD *)(v3 + 8);
          v6 = v5 + 32LL * *(unsigned int *)(v3 + 16);
          if ( v5 != v6 )
          {
            do
            {
              v6 -= 32LL;
              v7 = *(_QWORD *)(v6 + 8);
              if ( v7 != v6 + 24 )
                _libc_free(v7);
            }
            while ( v5 != v6 );
            v6 = *(_QWORD *)(v3 + 8);
          }
          if ( v6 != v3 + 24 )
            break;
        }
        v3 += 88;
        if ( v4 == v3 )
          goto LABEL_12;
      }
      v3 += 88;
      _libc_free(v6);
    }
    while ( v4 != v3 );
LABEL_12:
    v2 = *(unsigned int *)(a1 + 384);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 368), 88 * v2, 8);
  v8 = *(_QWORD *)(a1 + 288);
  if ( v8 != a1 + 304 )
    _libc_free(v8);
  v9 = *(_QWORD *)(a1 + 208);
  if ( v9 != a1 + 224 )
    _libc_free(v9);
  *(_QWORD *)a1 = &unk_49DAF80;
  sub_BB9100(a1);
  j_j___libc_free_0(a1);
}
