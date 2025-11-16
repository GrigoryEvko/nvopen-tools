// Function: sub_2351450
// Address: 0x2351450
//
void __fastcall sub_2351450(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5, unsigned __int64 a6)
{
  unsigned __int64 *v7; // r12
  unsigned __int64 v8; // r12
  __int64 v9; // rcx
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // r8
  __int64 v14; // r14
  __int64 v15; // r14
  __int64 v16; // rbx
  _QWORD *v17; // rdi
  __int64 v18; // r14
  unsigned __int64 v19; // r8
  __int64 v20; // r14
  __int64 v21; // rbx
  _QWORD *v22; // rdi
  unsigned __int64 v23; // r12

  v7 = (unsigned __int64 *)a1[5];
  *a1 = &unk_4A0D878;
  if ( v7 )
  {
    if ( (unsigned __int64 *)*v7 != v7 + 2 )
      _libc_free(*v7);
    a2 = 80;
    j_j___libc_free_0((unsigned __int64)v7);
  }
  v8 = a1[4];
  if ( v8 )
  {
    v9 = *(unsigned int *)(v8 + 376);
    if ( (_DWORD)v9 )
    {
      a2 = (__int64)sub_ED5FB0;
      sub_EDA800(v8 + 280, (char *)sub_ED5FB0, 0, v9, a5, a6);
    }
    *(_QWORD *)(v8 + 176) = 0;
    sub_B72320(v8 + 184, a2);
    v10 = *(_QWORD *)(v8 + 152);
    if ( v10 )
      j_j___libc_free_0(v10);
    sub_C7D6A0(*(_QWORD *)(v8 + 128), 16LL * *(unsigned int *)(v8 + 144), 8);
    v11 = *(_QWORD *)(v8 + 96);
    if ( v11 )
      j_j___libc_free_0(v11);
    v12 = *(_QWORD *)(v8 + 72);
    if ( v12 )
      j_j___libc_free_0(v12);
    v13 = *(_QWORD *)(v8 + 48);
    if ( *(_DWORD *)(v8 + 60) )
    {
      v14 = *(unsigned int *)(v8 + 56);
      if ( (_DWORD)v14 )
      {
        v15 = 8 * v14;
        v16 = 0;
        do
        {
          v17 = *(_QWORD **)(v13 + v16);
          if ( v17 != (_QWORD *)-8LL && v17 )
          {
            sub_C7D6A0((__int64)v17, *v17 + 9LL, 8);
            v13 = *(_QWORD *)(v8 + 48);
          }
          v16 += 8;
        }
        while ( v15 != v16 );
      }
    }
    _libc_free(v13);
    if ( *(_DWORD *)(v8 + 36) )
    {
      v18 = *(unsigned int *)(v8 + 32);
      v19 = *(_QWORD *)(v8 + 24);
      if ( (_DWORD)v18 )
      {
        v20 = 8 * v18;
        v21 = 0;
        do
        {
          v22 = *(_QWORD **)(v19 + v21);
          if ( v22 != (_QWORD *)-8LL && v22 )
          {
            sub_C7D6A0((__int64)v22, *v22 + 9LL, 8);
            v19 = *(_QWORD *)(v8 + 24);
          }
          v21 += 8;
        }
        while ( v20 != v21 );
      }
    }
    else
    {
      v19 = *(_QWORD *)(v8 + 24);
    }
    _libc_free(v19);
    j_j___libc_free_0(v8);
  }
  v23 = a1[2];
  if ( v23 )
  {
    sub_9CD560(a1[2]);
    j_j___libc_free_0(v23);
  }
}
