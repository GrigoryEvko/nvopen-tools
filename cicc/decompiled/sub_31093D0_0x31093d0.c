// Function: sub_31093D0
// Address: 0x31093d0
//
void __fastcall sub_31093D0(unsigned __int64 a1)
{
  int v2; // edx
  unsigned __int64 v3; // r8
  __int64 v4; // r13
  __int64 v5; // r13
  __int64 v6; // rbx
  _QWORD *v7; // rdi
  __int64 v8; // r13
  unsigned __int64 v9; // r8
  __int64 v10; // r13
  __int64 v11; // rbx
  _QWORD *v12; // rdi
  __int64 v13; // rdi

  v2 = *(_DWORD *)(a1 + 172);
  v3 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)a1 = &unk_4A329D8;
  if ( v2 )
  {
    v4 = *(unsigned int *)(a1 + 168);
    if ( (_DWORD)v4 )
    {
      v5 = 8 * v4;
      v6 = 0;
      do
      {
        v7 = *(_QWORD **)(v3 + v6);
        if ( v7 != (_QWORD *)-8LL && v7 )
        {
          sub_C7D6A0((__int64)v7, *v7 + 9LL, 8);
          v3 = *(_QWORD *)(a1 + 160);
        }
        v6 += 8;
      }
      while ( v5 != v6 );
    }
  }
  _libc_free(v3);
  if ( *(_DWORD *)(a1 + 148) )
  {
    v8 = *(unsigned int *)(a1 + 144);
    v9 = *(_QWORD *)(a1 + 136);
    if ( (_DWORD)v8 )
    {
      v10 = 8 * v8;
      v11 = 0;
      do
      {
        v12 = *(_QWORD **)(v9 + v11);
        if ( v12 != (_QWORD *)-8LL && v12 )
        {
          sub_C7D6A0((__int64)v12, *v12 + 17LL, 8);
          v9 = *(_QWORD *)(a1 + 136);
        }
        v11 += 8;
      }
      while ( v10 != v11 );
    }
  }
  else
  {
    v9 = *(_QWORD *)(a1 + 136);
  }
  _libc_free(v9);
  v13 = *(_QWORD *)(a1 + 80);
  if ( v13 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v13 + 8LL))(v13);
  sub_30CBA20((_QWORD *)a1);
  j_j___libc_free_0(a1);
}
