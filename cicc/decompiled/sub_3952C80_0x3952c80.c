// Function: sub_3952C80
// Address: 0x3952c80
//
void *__fastcall sub_3952C80(_QWORD *a1)
{
  unsigned __int64 v2; // r13
  unsigned __int64 v3; // rdi
  __int64 v4; // r14
  _QWORD *v5; // rbx
  _QWORD *v6; // r14
  unsigned __int64 v7; // r15
  unsigned __int64 v8; // rdi

  v2 = a1[20];
  *a1 = &unk_4A3F298;
  if ( v2 )
  {
    v3 = *(_QWORD *)(v2 + 192);
    if ( v3 != *(_QWORD *)(v2 + 184) )
      _libc_free(v3);
    j___libc_free_0(*(_QWORD *)(v2 + 152));
    v4 = *(unsigned int *)(v2 + 136);
    if ( (_DWORD)v4 )
    {
      v5 = *(_QWORD **)(v2 + 120);
      v6 = &v5[2 * v4];
      do
      {
        if ( *v5 != -16 && *v5 != -8 )
        {
          v7 = v5[1];
          if ( v7 )
          {
            _libc_free(*(_QWORD *)(v7 + 48));
            _libc_free(*(_QWORD *)(v7 + 24));
            j_j___libc_free_0(v7);
          }
        }
        v5 += 2;
      }
      while ( v6 != v5 );
    }
    j___libc_free_0(*(_QWORD *)(v2 + 120));
    v8 = *(_QWORD *)(v2 + 88);
    if ( v8 )
      j_j___libc_free_0(v8);
    j___libc_free_0(*(_QWORD *)(v2 + 64));
    j_j___libc_free_0(v2);
  }
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
