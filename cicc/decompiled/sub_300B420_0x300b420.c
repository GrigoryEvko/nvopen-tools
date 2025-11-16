// Function: sub_300B420
// Address: 0x300b420
//
void __fastcall sub_300B420(unsigned __int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r14
  __int64 v5; // r13
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdi

  *(_QWORD *)a1 = &unk_4A2DE80;
  v2 = *(unsigned int *)(a1 + 328);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD *)(a1 + 312);
    v4 = v3 + 72 * v2;
    do
    {
      while ( 1 )
      {
        v5 = v3 + 72;
        if ( *(_DWORD *)v3 <= 0xFFFFFFFD )
        {
          v6 = *(_QWORD *)(v3 + 56);
          if ( v6 != v5 )
            _libc_free(v6);
          v7 = *(_QWORD *)(v3 + 40);
          if ( v7 != v3 + 56 )
            break;
        }
        v3 += 72;
        if ( v4 == v5 )
          goto LABEL_9;
      }
      _libc_free(v7);
      v3 += 72;
    }
    while ( v4 != v5 );
LABEL_9:
    v2 = *(unsigned int *)(a1 + 328);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 312), 72 * v2, 8);
  v8 = *(_QWORD *)(a1 + 280);
  if ( v8 != a1 + 296 )
    _libc_free(v8);
  v9 = *(_QWORD *)(a1 + 256);
  if ( v9 != a1 + 272 )
    _libc_free(v9);
  v10 = *(_QWORD *)(a1 + 232);
  if ( v10 != a1 + 248 )
    _libc_free(v10);
  *(_QWORD *)a1 = &unk_49DAF80;
  sub_BB9100(a1);
  j_j___libc_free_0(a1);
}
