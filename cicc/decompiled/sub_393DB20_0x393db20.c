// Function: sub_393DB20
// Address: 0x393db20
//
void __fastcall sub_393DB20(unsigned __int64 a1)
{
  unsigned __int64 v1; // r13
  unsigned __int64 v2; // r14
  int v3; // eax
  __int64 v4; // r12
  unsigned __int64 v5; // r8
  __int64 v6; // r12
  __int64 v7; // rbx
  unsigned __int64 v8; // rdi

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      sub_393DB20(*(_QWORD *)(v1 + 24));
      v3 = *(_DWORD *)(v1 + 60);
      v1 = *(_QWORD *)(v1 + 16);
      if ( v3 )
      {
        v4 = *(unsigned int *)(v2 + 56);
        v5 = *(_QWORD *)(v2 + 48);
        if ( (_DWORD)v4 )
        {
          v6 = 8 * v4;
          v7 = 0;
          do
          {
            v8 = *(_QWORD *)(v5 + v7);
            if ( v8 != -8 )
            {
              if ( v8 )
              {
                _libc_free(v8);
                v5 = *(_QWORD *)(v2 + 48);
              }
            }
            v7 += 8;
          }
          while ( v6 != v7 );
        }
      }
      else
      {
        v5 = *(_QWORD *)(v2 + 48);
      }
      _libc_free(v5);
      j_j___libc_free_0(v2);
    }
    while ( v1 );
  }
}
