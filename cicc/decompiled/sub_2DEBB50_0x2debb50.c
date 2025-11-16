// Function: sub_2DEBB50
// Address: 0x2debb50
//
void __fastcall sub_2DEBB50(_QWORD *a1)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // r15
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // rbx
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // rdi

  v2 = a1[16];
  *a1 = off_49D4228;
  if ( v2 )
  {
    v3 = 152LL * *(_QWORD *)(v2 - 8);
    v4 = v2 + v3;
    if ( v2 != v2 + v3 )
    {
      do
      {
        v4 -= 152;
        if ( *(_DWORD *)(v4 + 136) > 0x40u )
        {
          v5 = *(_QWORD *)(v4 + 128);
          if ( v5 )
            j_j___libc_free_0_0(v5);
        }
        v6 = *(_QWORD *)(v4 + 16);
        v7 = v6 + 24LL * *(unsigned int *)(v4 + 24);
        if ( v6 != v7 )
        {
          do
          {
            v7 -= 24LL;
            if ( *(_DWORD *)(v7 + 16) > 0x40u )
            {
              v8 = *(_QWORD *)(v7 + 8);
              if ( v8 )
                j_j___libc_free_0_0(v8);
            }
          }
          while ( v6 != v7 );
          v6 = *(_QWORD *)(v4 + 16);
        }
        if ( v6 != v4 + 32 )
          _libc_free(v6);
      }
      while ( a1[16] != v4 );
    }
    j_j_j___libc_free_0_0(v4 - 8);
  }
  v9 = a1[11];
  while ( v9 )
  {
    sub_2DEAE80(*(_QWORD *)(v9 + 24));
    v10 = v9;
    v9 = *(_QWORD *)(v9 + 16);
    j_j___libc_free_0(v10);
  }
  v11 = a1[5];
  while ( v11 )
  {
    sub_2DEACB0(*(_QWORD *)(v11 + 24));
    v12 = v11;
    v11 = *(_QWORD *)(v11 + 16);
    j_j___libc_free_0(v12);
  }
  j_j___libc_free_0((unsigned __int64)a1);
}
