// Function: sub_1DF3AD0
// Address: 0x1df3ad0
//
void *__fastcall sub_1DF3AD0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r13
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi

  *(_QWORD *)a1 = off_49FB508;
  v2 = *(unsigned int *)(a1 + 504);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD *)(a1 + 488);
    v4 = v3 + 40 * v2;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v3 <= 0xFFFFFFFD )
        {
          v5 = *(_QWORD *)(v3 + 8);
          if ( v5 != v3 + 24 )
            break;
        }
        v3 += 40;
        if ( v4 == v3 )
          goto LABEL_7;
      }
      _libc_free(v5);
      v3 += 40;
    }
    while ( v4 != v3 );
  }
LABEL_7:
  j___libc_free_0(*(_QWORD *)(a1 + 488));
  j___libc_free_0(*(_QWORD *)(a1 + 456));
  j___libc_free_0(*(_QWORD *)(a1 + 424));
  v6 = *(_QWORD *)(a1 + 336);
  if ( v6 != a1 + 352 )
    _libc_free(v6);
  if ( (*(_BYTE *)(a1 + 264) & 1) == 0 )
    j___libc_free_0(*(_QWORD *)(a1 + 272));
  _libc_free(*(_QWORD *)(a1 + 208));
  _libc_free(*(_QWORD *)(a1 + 184));
  _libc_free(*(_QWORD *)(a1 + 160));
  *(_QWORD *)a1 = &unk_49EE078;
  return sub_16366C0((_QWORD *)a1);
}
