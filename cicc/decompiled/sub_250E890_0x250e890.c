// Function: sub_250E890
// Address: 0x250e890
//
__int64 __fastcall sub_250E890(__int64 a1)
{
  unsigned __int64 v2; // rdi
  _DWORD *v4; // rax
  _DWORD *v5; // r12
  _DWORD *v6; // rbx
  unsigned __int64 *v7; // rax

  if ( *(_DWORD *)(a1 + 16) )
  {
    v4 = *(_DWORD **)(a1 + 8);
    v5 = &v4[4 * *(unsigned int *)(a1 + 24)];
    if ( v4 != v5 )
    {
      while ( 1 )
      {
        v6 = v4;
        if ( *v4 <= 0xFFFFFFFD )
          break;
        v4 += 4;
        if ( v5 == v4 )
          goto LABEL_2;
      }
      while ( v5 != v6 )
      {
        v7 = (unsigned __int64 *)*((_QWORD *)v6 + 1);
        if ( (unsigned __int64 *)*v7 != v7 + 2 )
          _libc_free(*v7);
        v6 += 4;
        if ( v6 == v5 )
          break;
        while ( *v6 > 0xFFFFFFFD )
        {
          v6 += 4;
          if ( v5 == v6 )
            goto LABEL_2;
        }
      }
    }
  }
LABEL_2:
  v2 = *(_QWORD *)(a1 + 32);
  if ( v2 != a1 + 48 )
    _libc_free(v2);
  return sub_C7D6A0(*(_QWORD *)(a1 + 8), 16LL * *(unsigned int *)(a1 + 24), 8);
}
