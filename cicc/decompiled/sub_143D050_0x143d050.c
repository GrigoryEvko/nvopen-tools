// Function: sub_143D050
// Address: 0x143d050
//
__int64 __fastcall sub_143D050(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r13
  unsigned __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r13
  unsigned __int64 v11; // rdi

  v2 = sub_22077B0(112);
  if ( v2 )
  {
    *(_DWORD *)v2 = 1;
    *(_QWORD *)(v2 + 8) = 0;
    *(_QWORD *)(v2 + 16) = 0;
    *(_QWORD *)(v2 + 24) = 0;
    *(_DWORD *)(v2 + 32) = 0;
    *(_QWORD *)(v2 + 40) = 0;
    *(_QWORD *)(v2 + 48) = 0;
    *(_QWORD *)(v2 + 56) = 0;
    *(_DWORD *)(v2 + 64) = 0;
    *(_QWORD *)(v2 + 72) = 0;
    *(_QWORD *)(v2 + 80) = 0;
    *(_QWORD *)(v2 + 88) = 0;
    *(_DWORD *)(v2 + 96) = 0;
    *(_QWORD *)(v2 + 104) = a2;
  }
  v3 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 160) = v2;
  if ( v3 )
  {
    v4 = *(unsigned int *)(v3 + 96);
    if ( (_DWORD)v4 )
    {
      v5 = *(_QWORD *)(v3 + 80);
      v6 = v5 + 80 * v4;
      do
      {
        while ( 1 )
        {
          if ( *(_DWORD *)v5 <= 0xFFFFFFFD )
          {
            v7 = *(_QWORD *)(v5 + 24);
            if ( v7 != *(_QWORD *)(v5 + 16) )
              break;
          }
          v5 += 80;
          if ( v6 == v5 )
            goto LABEL_10;
        }
        _libc_free(v7);
        v5 += 80;
      }
      while ( v6 != v5 );
    }
LABEL_10:
    j___libc_free_0(*(_QWORD *)(v3 + 80));
    v8 = *(unsigned int *)(v3 + 64);
    if ( (_DWORD)v8 )
    {
      v9 = *(_QWORD *)(v3 + 48);
      v10 = v9 + 80 * v8;
      do
      {
        while ( 1 )
        {
          if ( *(_DWORD *)v9 <= 0xFFFFFFFD )
          {
            v11 = *(_QWORD *)(v9 + 24);
            if ( v11 != *(_QWORD *)(v9 + 16) )
              break;
          }
          v9 += 80;
          if ( v10 == v9 )
            goto LABEL_16;
        }
        _libc_free(v11);
        v9 += 80;
      }
      while ( v10 != v9 );
    }
LABEL_16:
    j___libc_free_0(*(_QWORD *)(v3 + 48));
    j___libc_free_0(*(_QWORD *)(v3 + 16));
    j_j___libc_free_0(v3, 112);
  }
  return 0;
}
