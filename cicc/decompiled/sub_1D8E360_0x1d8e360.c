// Function: sub_1D8E360
// Address: 0x1d8e360
//
void *__fastcall sub_1D8E360(__int64 a1)
{
  _QWORD **v2; // rbx
  _QWORD **v3; // r12
  __int64 v4; // r13
  unsigned __int64 v5; // r8
  __int64 v6; // rbx
  __int64 v7; // rbx
  __int64 v8; // r12
  unsigned __int64 v9; // rdi
  __int64 v10; // rbx
  unsigned __int64 v11; // r13
  _QWORD *v12; // rdi
  _QWORD *v13; // r12
  __int64 (__fastcall *v14)(_QWORD *); // rax

  *(_QWORD *)a1 = &unk_49FA5D0;
  j___libc_free_0(*(_QWORD *)(a1 + 248));
  v2 = *(_QWORD ***)(a1 + 224);
  v3 = *(_QWORD ***)(a1 + 216);
  if ( v2 != v3 )
  {
    do
    {
      v4 = (__int64)*v3;
      if ( *v3 )
      {
        sub_1D8E2D0(*v3);
        j_j___libc_free_0(v4, 72);
      }
      ++v3;
    }
    while ( v2 != v3 );
    v3 = *(_QWORD ***)(a1 + 216);
  }
  if ( v3 )
    j_j___libc_free_0(v3, *(_QWORD *)(a1 + 232) - (_QWORD)v3);
  v5 = *(_QWORD *)(a1 + 184);
  if ( *(_DWORD *)(a1 + 196) )
  {
    v6 = *(unsigned int *)(a1 + 192);
    if ( (_DWORD)v6 )
    {
      v7 = 8 * v6;
      v8 = 0;
      do
      {
        v9 = *(_QWORD *)(v5 + v8);
        if ( v9 != -8 && v9 )
        {
          _libc_free(v9);
          v5 = *(_QWORD *)(a1 + 184);
        }
        v8 += 8;
      }
      while ( v7 != v8 );
    }
  }
  _libc_free(v5);
  v10 = *(_QWORD *)(a1 + 160);
  v11 = v10 + 8LL * *(unsigned int *)(a1 + 168);
  if ( v10 != v11 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v13 = *(_QWORD **)(v11 - 8);
        v11 -= 8LL;
        if ( v13 )
          break;
LABEL_20:
        if ( v10 == v11 )
          goto LABEL_24;
      }
      v14 = *(__int64 (__fastcall **)(_QWORD *))(*v13 + 8LL);
      if ( v14 == sub_1D59FF0 )
      {
        v12 = (_QWORD *)v13[1];
        *v13 = &unk_49F9CF0;
        if ( v12 != v13 + 3 )
          j_j___libc_free_0(v12, v13[3] + 1LL);
        j_j___libc_free_0(v13, 56);
        goto LABEL_20;
      }
      v14(v13);
      if ( v10 == v11 )
      {
LABEL_24:
        v11 = *(_QWORD *)(a1 + 160);
        break;
      }
    }
  }
  if ( v11 != a1 + 176 )
    _libc_free(v11);
  return sub_16367B0((_QWORD *)a1);
}
