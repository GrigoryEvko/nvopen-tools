// Function: sub_21EA2C0
// Address: 0x21ea2c0
//
void *__fastcall sub_21EA2C0(_QWORD *a1)
{
  __int64 v2; // r13
  __int64 v3; // r14
  _QWORD *v4; // rbx
  _QWORD *v5; // r14
  __int64 v6; // r15
  __int64 v7; // rdi

  v2 = a1[29];
  *a1 = &unk_4A04018;
  if ( v2 )
  {
    j___libc_free_0(*(_QWORD *)(v2 + 304));
    j___libc_free_0(*(_QWORD *)(v2 + 272));
    j___libc_free_0(*(_QWORD *)(v2 + 240));
    j___libc_free_0(*(_QWORD *)(v2 + 208));
    j___libc_free_0(*(_QWORD *)(v2 + 152));
    v3 = *(unsigned int *)(v2 + 136);
    if ( (_DWORD)v3 )
    {
      v4 = *(_QWORD **)(v2 + 120);
      v5 = &v4[2 * v3];
      do
      {
        if ( *v4 != -16 && *v4 != -8 )
        {
          v6 = v4[1];
          if ( v6 )
          {
            _libc_free(*(_QWORD *)(v6 + 48));
            _libc_free(*(_QWORD *)(v6 + 24));
            j_j___libc_free_0(v6, 72);
          }
        }
        v4 += 2;
      }
      while ( v5 != v4 );
    }
    j___libc_free_0(*(_QWORD *)(v2 + 120));
    v7 = *(_QWORD *)(v2 + 88);
    if ( v7 )
      j_j___libc_free_0(v7, *(_QWORD *)(v2 + 104) - v7);
    j___libc_free_0(*(_QWORD *)(v2 + 64));
    j_j___libc_free_0(v2, 328);
  }
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
