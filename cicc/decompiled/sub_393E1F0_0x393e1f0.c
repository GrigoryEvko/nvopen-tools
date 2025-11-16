// Function: sub_393E1F0
// Address: 0x393e1f0
//
void __fastcall sub_393E1F0(__int64 a1)
{
  unsigned __int64 v2; // r12
  unsigned __int64 v3; // rdi
  __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  __int64 v6; // r15
  __int64 v7; // r12
  unsigned __int64 v8; // r15
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // r13
  _QWORD *v11; // rdi
  __int64 v12; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 56);
  *(_QWORD *)a1 = &unk_4A3F020;
  if ( v2 )
  {
    v3 = *(_QWORD *)(v2 + 8);
    if ( v3 )
      j_j___libc_free_0(v3);
    j_j___libc_free_0(v2);
  }
  v4 = *(_QWORD *)(a1 + 48);
  if ( v4 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v4 + 8LL))(v4);
  v5 = *(_QWORD *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 20) )
  {
    v6 = *(unsigned int *)(a1 + 16);
    if ( (_DWORD)v6 )
    {
      v7 = 0;
      v12 = 8 * v6;
      do
      {
        v8 = *(_QWORD *)(v5 + v7);
        if ( v8 != -8 && v8 )
        {
          v9 = *(_QWORD *)(v8 + 104);
          while ( v9 )
          {
            v10 = v9;
            sub_393DEF0(*(_QWORD **)(v9 + 24));
            v11 = *(_QWORD **)(v9 + 56);
            v9 = *(_QWORD *)(v9 + 16);
            sub_393E140(v11);
            j_j___libc_free_0(v10);
          }
          sub_393DB20(*(_QWORD *)(v8 + 56));
          _libc_free(v8);
          v5 = *(_QWORD *)(a1 + 8);
        }
        v7 += 8;
      }
      while ( v12 != v7 );
    }
  }
  _libc_free(v5);
}
