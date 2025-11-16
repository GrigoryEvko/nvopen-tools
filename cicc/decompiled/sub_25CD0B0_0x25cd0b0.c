// Function: sub_25CD0B0
// Address: 0x25cd0b0
//
void __fastcall sub_25CD0B0(unsigned __int64 a1)
{
  int v2; // eax
  unsigned __int64 v3; // rdi
  __int64 v4; // r14
  __int64 v5; // r14
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 v8; // r15

  *(_QWORD *)a1 = &unk_4A1F2E8;
  v2 = *(_DWORD *)(a1 + 52);
  v3 = *(_QWORD *)(a1 + 40);
  if ( v2 )
  {
    v4 = *(unsigned int *)(a1 + 48);
    if ( (_DWORD)v4 )
    {
      v5 = 8 * v4;
      v6 = 0;
      do
      {
        v7 = *(_QWORD *)(v3 + v6);
        if ( v7 != -8 && v7 )
        {
          v8 = *(_QWORD *)v7 + 41LL;
          sub_C7D6A0(*(_QWORD *)(v7 + 16), 8LL * *(unsigned int *)(v7 + 32), 8);
          sub_C7D6A0(v7, v8, 8);
          v3 = *(_QWORD *)(a1 + 40);
        }
        v6 += 8;
      }
      while ( v5 != v6 );
    }
  }
  _libc_free(v3);
  j_j___libc_free_0(a1);
}
