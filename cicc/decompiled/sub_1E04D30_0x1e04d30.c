// Function: sub_1E04D30
// Address: 0x1e04d30
//
__int64 __fastcall sub_1E04D30(_QWORD *a1)
{
  __int64 v2; // r14
  __int64 v3; // r13
  _QWORD *v4; // rbx
  _QWORD *v5; // r13
  __int64 v6; // r15
  __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi

  v2 = a1[164];
  *a1 = &unk_49FB698;
  if ( v2 )
  {
    v3 = *(unsigned int *)(v2 + 48);
    if ( (_DWORD)v3 )
    {
      v4 = *(_QWORD **)(v2 + 32);
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
    j___libc_free_0(*(_QWORD *)(v2 + 32));
    if ( *(_QWORD *)v2 != v2 + 16 )
      _libc_free(*(_QWORD *)v2);
    j_j___libc_free_0(v2, 80);
  }
  v8 = a1[129];
  if ( v8 != a1[128] )
    _libc_free(v8);
  v9 = a1[29];
  if ( (_QWORD *)v9 != a1 + 31 )
    _libc_free(v9);
  _libc_free(a1[26]);
  _libc_free(a1[23]);
  _libc_free(a1[20]);
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 1320);
}
