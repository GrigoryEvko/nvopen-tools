// Function: sub_1BBA840
// Address: 0x1bba840
//
__int64 __fastcall sub_1BBA840(_QWORD *a1)
{
  __int64 v2; // rbx
  __int64 v3; // r15
  _QWORD *v4; // r14
  _QWORD *v5; // r12
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r12
  unsigned __int64 v9; // rdi

  v2 = a1[41];
  v3 = a1[40];
  *a1 = off_49F7068;
  if ( v2 != v3 )
  {
    do
    {
      v4 = *(_QWORD **)(v3 + 8);
      v5 = &v4[3 * *(unsigned int *)(v3 + 16)];
      if ( v4 != v5 )
      {
        do
        {
          v6 = *(v5 - 1);
          v5 -= 3;
          if ( v6 != 0 && v6 != -8 && v6 != -16 )
            sub_1649B30(v5);
        }
        while ( v4 != v5 );
        v5 = *(_QWORD **)(v3 + 8);
      }
      if ( v5 != (_QWORD *)(v3 + 24) )
        _libc_free((unsigned __int64)v5);
      v3 += 216;
    }
    while ( v2 != v3 );
    v3 = a1[40];
  }
  if ( v3 )
    j_j___libc_free_0(v3, a1[42] - v3);
  j___libc_free_0(a1[37]);
  v7 = a1[34];
  v8 = a1[33];
  if ( v7 != v8 )
  {
    do
    {
      v9 = *(_QWORD *)(v8 + 8);
      if ( v9 != v8 + 24 )
        _libc_free(v9);
      v8 += 88;
    }
    while ( v7 != v8 );
    v8 = a1[33];
  }
  if ( v8 )
    j_j___libc_free_0(v8, a1[35] - v8);
  j___libc_free_0(a1[30]);
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 344);
}
