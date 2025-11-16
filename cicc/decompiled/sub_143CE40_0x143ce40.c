// Function: sub_143CE40
// Address: 0x143ce40
//
__int64 __fastcall sub_143CE40(_QWORD *a1)
{
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r14
  unsigned __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // r14
  unsigned __int64 v10; // rdi

  v2 = a1[20];
  *a1 = &unk_49EB8D8;
  if ( v2 )
  {
    v3 = *(unsigned int *)(v2 + 96);
    if ( (_DWORD)v3 )
    {
      v4 = *(_QWORD *)(v2 + 80);
      v5 = v4 + 80 * v3;
      do
      {
        while ( 1 )
        {
          if ( *(_DWORD *)v4 <= 0xFFFFFFFD )
          {
            v6 = *(_QWORD *)(v4 + 24);
            if ( v6 != *(_QWORD *)(v4 + 16) )
              break;
          }
          v4 += 80;
          if ( v5 == v4 )
            goto LABEL_8;
        }
        _libc_free(v6);
        v4 += 80;
      }
      while ( v5 != v4 );
    }
LABEL_8:
    j___libc_free_0(*(_QWORD *)(v2 + 80));
    v7 = *(unsigned int *)(v2 + 64);
    if ( (_DWORD)v7 )
    {
      v8 = *(_QWORD *)(v2 + 48);
      v9 = v8 + 80 * v7;
      do
      {
        while ( 1 )
        {
          if ( *(_DWORD *)v8 <= 0xFFFFFFFD )
          {
            v10 = *(_QWORD *)(v8 + 24);
            if ( v10 != *(_QWORD *)(v8 + 16) )
              break;
          }
          v8 += 80;
          if ( v9 == v8 )
            goto LABEL_14;
        }
        _libc_free(v10);
        v8 += 80;
      }
      while ( v9 != v8 );
    }
LABEL_14:
    j___libc_free_0(*(_QWORD *)(v2 + 48));
    j___libc_free_0(*(_QWORD *)(v2 + 16));
    j_j___libc_free_0(v2, 112);
  }
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
