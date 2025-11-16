// Function: sub_2FD4CE0
// Address: 0x2fd4ce0
//
void __fastcall sub_2FD4CE0(unsigned __int64 a1)
{
  unsigned __int64 v2; // r12
  __int64 v3; // rsi
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi

  v2 = *(_QWORD *)(a1 + 376);
  *(_QWORD *)a1 = off_4A2C550;
  if ( v2 )
  {
    sub_C7D6A0(*(_QWORD *)(v2 + 16), 16LL * *(unsigned int *)(v2 + 32), 8);
    j_j___libc_free_0(v2);
  }
  v3 = *(unsigned int *)(a1 + 368);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD *)(a1 + 352);
    v5 = v4 + 32 * v3;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v4 <= 0xFFFFFFFD )
        {
          v6 = *(_QWORD *)(v4 + 8);
          if ( v6 )
            break;
        }
        v4 += 32;
        if ( v5 == v4 )
          goto LABEL_9;
      }
      v4 += 32;
      j_j___libc_free_0(v6);
    }
    while ( v5 != v4 );
LABEL_9:
    v3 = *(unsigned int *)(a1 + 368);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 352), 32 * v3, 8);
  v7 = *(_QWORD *)(a1 + 264);
  if ( v7 != a1 + 280 )
    _libc_free(v7);
  *(_QWORD *)a1 = &unk_49DAF80;
  sub_BB9100(a1);
  j_j___libc_free_0(a1);
}
