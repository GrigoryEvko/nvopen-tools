// Function: sub_C62F20
// Address: 0xc62f20
//
void __fastcall sub_C62F20(_BYTE *a1, char *a2)
{
  void (__fastcall *v3)(_BYTE *, _BYTE *, __int64); // rax
  _BYTE *v4; // rdi
  void (__fastcall *v5)(_BYTE *, _BYTE *, __int64); // rax
  _BYTE *v6; // rdi
  void (__fastcall *v7)(_BYTE *, _BYTE *, __int64); // rax
  __int64 v8; // rdi
  _QWORD *v9; // r13
  _QWORD *v10; // r12
  _QWORD *v11; // rdi
  _BYTE *v12; // rdi
  _QWORD *v13; // r13
  _QWORD *v14; // r12
  __int64 v15; // r12
  __int64 v16; // r13
  __int64 v17; // rdi
  __int64 v18; // rsi
  __int64 v19; // r12
  __int64 v20; // rsi
  __int64 v21; // r13
  __int64 v22; // rdi
  __int64 v23; // rdi
  __int64 v24; // rax

  if ( a1[105] )
  {
    a2 = (char *)sub_C5F790((__int64)a1, (__int64)a2);
    sub_C626F0((__int64)a1, (__int64)a2);
  }
  *((_QWORD *)a1 + 69) = &unk_49D9AD8;
  v3 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))*((_QWORD *)a1 + 92);
  if ( v3 )
  {
    a2 = a1 + 720;
    v3(a1 + 720, a1 + 720, 3);
  }
  if ( !a1[676] )
    _libc_free(*((_QWORD *)a1 + 82), a2);
  v4 = (_BYTE *)*((_QWORD *)a1 + 78);
  if ( v4 != a1 + 640 )
    _libc_free(v4, a2);
  *((_QWORD *)a1 + 44) = &unk_49D9AD8;
  v5 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))*((_QWORD *)a1 + 67);
  if ( v5 )
  {
    a2 = a1 + 520;
    v5(a1 + 520, a1 + 520, 3);
  }
  if ( !a1[476] )
    _libc_free(*((_QWORD *)a1 + 57), a2);
  v6 = (_BYTE *)*((_QWORD *)a1 + 53);
  if ( v6 != a1 + 440 )
    _libc_free(v6, a2);
  *((_QWORD *)a1 + 14) = &unk_49DC600;
  v7 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))*((_QWORD *)a1 + 42);
  if ( v7 )
  {
    a2 = a1 + 320;
    v7(a1 + 320, a1 + 320, 3);
  }
  v8 = *((_QWORD *)a1 + 36);
  if ( v8 )
  {
    a2 = (char *)(*((_QWORD *)a1 + 38) - v8);
    j_j___libc_free_0(v8, a2);
  }
  v9 = (_QWORD *)*((_QWORD *)a1 + 33);
  v10 = (_QWORD *)*((_QWORD *)a1 + 32);
  if ( v9 != v10 )
  {
    do
    {
      v11 = (_QWORD *)v10[1];
      *v10 = &unk_49DACE8;
      if ( v11 != v10 + 3 )
      {
        a2 = (char *)(v10[3] + 1LL);
        j_j___libc_free_0(v11, a2);
      }
      v10 += 6;
    }
    while ( v9 != v10 );
    v10 = (_QWORD *)*((_QWORD *)a1 + 32);
  }
  if ( v10 )
  {
    a2 = (char *)(*((_QWORD *)a1 + 34) - (_QWORD)v10);
    j_j___libc_free_0(v10, a2);
  }
  if ( !a1[236] )
    _libc_free(*((_QWORD *)a1 + 27), a2);
  v12 = (_BYTE *)*((_QWORD *)a1 + 23);
  if ( v12 != a1 + 200 )
    _libc_free(v12, a2);
  v13 = (_QWORD *)*((_QWORD *)a1 + 11);
  v14 = (_QWORD *)*((_QWORD *)a1 + 10);
  if ( v13 != v14 )
  {
    do
    {
      if ( (_QWORD *)*v14 != v14 + 2 )
        j_j___libc_free_0(*v14, v14[2] + 1LL);
      v14 += 4;
    }
    while ( v13 != v14 );
    v14 = (_QWORD *)*((_QWORD *)a1 + 10);
  }
  if ( v14 )
    j_j___libc_free_0(v14, *((_QWORD *)a1 + 12) - (_QWORD)v14);
  v15 = *((_QWORD *)a1 + 6);
  while ( v15 )
  {
    v16 = v15;
    sub_C600E0(*(_QWORD **)(v15 + 24));
    v17 = *(_QWORD *)(v15 + 32);
    v15 = *(_QWORD *)(v15 + 16);
    if ( v17 != v16 + 48 )
      j_j___libc_free_0(v17, *(_QWORD *)(v16 + 48) + 1LL);
    j_j___libc_free_0(v16, 72);
  }
  v18 = *((unsigned int *)a1 + 6);
  if ( (_DWORD)v18 )
  {
    v19 = *((_QWORD *)a1 + 1);
    v20 = v18 << 7;
    v21 = v19 + v20;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v19 <= 0xFFFFFFFD )
        {
          v22 = *(_QWORD *)(v19 + 64);
          if ( v22 != v19 + 80 )
            _libc_free(v22, v20);
          v23 = *(_QWORD *)(v19 + 32);
          if ( v23 != v19 + 48 )
            break;
        }
        v19 += 128;
        if ( v21 == v19 )
          goto LABEL_49;
      }
      v24 = *(_QWORD *)(v19 + 48);
      v19 += 128;
      v20 = v24 + 1;
      j_j___libc_free_0(v23, v24 + 1);
    }
    while ( v21 != v19 );
LABEL_49:
    v18 = *((unsigned int *)a1 + 6);
  }
  sub_C7D6A0(*((_QWORD *)a1 + 1), v18 << 7, 8);
}
