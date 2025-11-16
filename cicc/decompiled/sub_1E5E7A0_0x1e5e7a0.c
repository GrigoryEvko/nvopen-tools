// Function: sub_1E5E7A0
// Address: 0x1e5e7a0
//
void *__fastcall sub_1E5E7A0(_QWORD *a1)
{
  __int64 v2; // r14
  __int64 v3; // r13
  _QWORD *v4; // rbx
  _QWORD *v5; // r13
  __int64 v6; // r15
  __int64 v7; // rdi

  v2 = a1[29];
  *a1 = &unk_49FC278;
  if ( v2 )
  {
    v3 = *(unsigned int *)(v2 + 72);
    if ( (_DWORD)v3 )
    {
      v4 = *(_QWORD **)(v2 + 56);
      v5 = &v4[2 * v3];
      do
      {
        if ( *v4 != -16 && *v4 != -8 )
        {
          v6 = v4[1];
          if ( v6 )
          {
            v7 = *(_QWORD *)(v6 + 24);
            if ( v7 )
              j_j___libc_free_0(v7, *(_QWORD *)(v6 + 40) - v7);
            j_j___libc_free_0(v6, 56);
          }
        }
        v4 += 2;
      }
      while ( v5 != v4 );
    }
    j___libc_free_0(*(_QWORD *)(v2 + 56));
    if ( *(_QWORD *)v2 != v2 + 16 )
      _libc_free(*(_QWORD *)v2);
    j_j___libc_free_0(v2, 104);
  }
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
