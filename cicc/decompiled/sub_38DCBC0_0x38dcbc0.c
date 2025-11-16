// Function: sub_38DCBC0
// Address: 0x38dcbc0
//
void __fastcall sub_38DCBC0(_QWORD *a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 *v3; // rbx
  unsigned __int64 *v4; // r12
  unsigned __int64 v5; // r13
  unsigned __int64 v6; // rdi
  __int64 v7; // r15
  unsigned __int64 v8; // r13
  __int64 v9; // rbx
  unsigned __int64 v10; // r12
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // r12
  void (__fastcall *v13)(unsigned __int64); // rax

  *a1 = &unk_4A3E670;
  v2 = a1[14];
  if ( (_QWORD *)v2 != a1 + 16 )
    _libc_free(v2);
  j___libc_free_0(a1[11]);
  v3 = (unsigned __int64 *)a1[7];
  v4 = (unsigned __int64 *)a1[6];
  if ( v3 != v4 )
  {
    do
    {
      v5 = *v4;
      if ( *v4 )
      {
        v6 = *(_QWORD *)(v5 + 72);
        if ( v6 )
          j_j___libc_free_0(v6);
        j_j___libc_free_0(v5);
      }
      ++v4;
    }
    while ( v3 != v4 );
    v4 = (unsigned __int64 *)a1[6];
  }
  if ( v4 )
    j_j___libc_free_0((unsigned __int64)v4);
  v7 = a1[4];
  v8 = a1[3];
  if ( v7 != v8 )
  {
    do
    {
      v9 = *(_QWORD *)(v8 + 40);
      v10 = *(_QWORD *)(v8 + 32);
      if ( v9 != v10 )
      {
        do
        {
          v11 = *(_QWORD *)(v10 + 24);
          if ( v11 )
            j_j___libc_free_0(v11);
          v10 += 48LL;
        }
        while ( v9 != v10 );
        v10 = *(_QWORD *)(v8 + 32);
      }
      if ( v10 )
        j_j___libc_free_0(v10);
      v8 += 80LL;
    }
    while ( v7 != v8 );
    v8 = a1[3];
  }
  if ( v8 )
    j_j___libc_free_0(v8);
  v12 = a1[2];
  if ( v12 )
  {
    v13 = *(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v12 + 8LL);
    if ( v13 == sub_38DBD30 )
    {
      nullsub_1936();
      j_j___libc_free_0(v12);
    }
    else
    {
      v13(a1[2]);
    }
  }
}
