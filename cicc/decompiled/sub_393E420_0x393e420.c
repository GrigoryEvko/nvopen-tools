// Function: sub_393E420
// Address: 0x393e420
//
void __fastcall sub_393E420(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi
  __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  __int64 v7; // r15
  __int64 v8; // r12
  unsigned __int64 v9; // r15
  unsigned __int64 v10; // rbx
  unsigned __int64 v11; // r13
  _QWORD *v12; // rdi
  __int64 v13; // [rsp+8h] [rbp-38h]

  *(_QWORD *)a1 = &unk_4A3F080;
  v2 = *(_QWORD *)(a1 + 88);
  if ( v2 )
    j_j___libc_free_0(v2);
  v3 = *(_QWORD *)(a1 + 56);
  *(_QWORD *)a1 = &unk_4A3F020;
  if ( v3 )
  {
    v4 = *(_QWORD *)(v3 + 8);
    if ( v4 )
      j_j___libc_free_0(v4);
    j_j___libc_free_0(v3);
  }
  v5 = *(_QWORD *)(a1 + 48);
  if ( v5 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v5 + 8LL))(v5);
  v6 = *(_QWORD *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 20) )
  {
    v7 = *(unsigned int *)(a1 + 16);
    if ( (_DWORD)v7 )
    {
      v8 = 0;
      v13 = 8 * v7;
      do
      {
        v9 = *(_QWORD *)(v6 + v8);
        if ( v9 && v9 != -8 )
        {
          v10 = *(_QWORD *)(v9 + 104);
          while ( v10 )
          {
            v11 = v10;
            sub_393DEF0(*(_QWORD **)(v10 + 24));
            v12 = *(_QWORD **)(v10 + 56);
            v10 = *(_QWORD *)(v10 + 16);
            sub_393E140(v12);
            j_j___libc_free_0(v11);
          }
          sub_393DB20(*(_QWORD *)(v9 + 56));
          _libc_free(v9);
          v6 = *(_QWORD *)(a1 + 8);
        }
        v8 += 8;
      }
      while ( v13 != v8 );
    }
  }
  _libc_free(v6);
}
