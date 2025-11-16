// Function: sub_393E670
// Address: 0x393e670
//
void __fastcall sub_393E670(__int64 a1)
{
  unsigned __int64 *v2; // r13
  unsigned __int64 *v3; // r12
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rdi
  __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  __int64 v8; // r15
  __int64 v9; // r13
  unsigned __int64 v10; // r15
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // r14
  _QWORD *v13; // rdi
  __int64 v14; // [rsp+8h] [rbp-38h]

  v2 = *(unsigned __int64 **)(a1 + 96);
  v3 = *(unsigned __int64 **)(a1 + 88);
  *(_QWORD *)a1 = &unk_4A3F110;
  if ( v2 != v3 )
  {
    do
    {
      if ( (unsigned __int64 *)*v3 != v3 + 2 )
        j_j___libc_free_0(*v3);
      v3 += 4;
    }
    while ( v2 != v3 );
    v3 = *(unsigned __int64 **)(a1 + 88);
  }
  if ( v3 )
    j_j___libc_free_0((unsigned __int64)v3);
  v4 = *(_QWORD *)(a1 + 56);
  *(_QWORD *)a1 = &unk_4A3F020;
  if ( v4 )
  {
    v5 = *(_QWORD *)(v4 + 8);
    if ( v5 )
      j_j___libc_free_0(v5);
    j_j___libc_free_0(v4);
  }
  v6 = *(_QWORD *)(a1 + 48);
  if ( v6 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL))(v6);
  v7 = *(_QWORD *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 20) )
  {
    v8 = *(unsigned int *)(a1 + 16);
    if ( (_DWORD)v8 )
    {
      v9 = 0;
      v14 = 8 * v8;
      do
      {
        v10 = *(_QWORD *)(v7 + v9);
        if ( v10 != -8 && v10 )
        {
          v11 = *(_QWORD *)(v10 + 104);
          while ( v11 )
          {
            v12 = v11;
            sub_393DEF0(*(_QWORD **)(v11 + 24));
            v13 = *(_QWORD **)(v11 + 56);
            v11 = *(_QWORD *)(v11 + 16);
            sub_393E140(v13);
            j_j___libc_free_0(v12);
          }
          sub_393DB20(*(_QWORD *)(v10 + 56));
          _libc_free(v10);
          v7 = *(_QWORD *)(a1 + 8);
        }
        v9 += 8;
      }
      while ( v14 != v9 );
    }
  }
  _libc_free(v7);
}
