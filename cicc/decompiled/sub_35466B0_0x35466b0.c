// Function: sub_35466B0
// Address: 0x35466b0
//
void __fastcall sub_35466B0(__int64 a1)
{
  __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  __int64 v4; // rax
  _QWORD *v5; // rbx
  _QWORD *v6; // r13
  unsigned __int64 v7; // rdi
  __int64 v8; // rax
  _QWORD *v9; // rbx
  _QWORD *v10; // r13
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi

  v2 = *(_QWORD *)(a1 + 528);
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  v3 = *(_QWORD *)(a1 + 480);
  if ( v3 != a1 + 496 )
    _libc_free(v3);
  sub_3546630((unsigned __int64 *)(a1 + 400));
  sub_3546630((unsigned __int64 *)(a1 + 320));
  sub_C7D6A0(*(_QWORD *)(a1 + 296), 24LL * *(unsigned int *)(a1 + 312), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 264), 16LL * *(unsigned int *)(a1 + 280), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 232), 16LL * *(unsigned int *)(a1 + 248), 8);
  v4 = *(unsigned int *)(a1 + 216);
  if ( (_DWORD)v4 )
  {
    v5 = *(_QWORD **)(a1 + 200);
    v6 = &v5[10 * v4];
    do
    {
      if ( *v5 != -8192 && *v5 != -4096 )
      {
        v7 = v5[1];
        if ( (_QWORD *)v7 != v5 + 3 )
          _libc_free(v7);
      }
      v5 += 10;
    }
    while ( v6 != v5 );
    v4 = *(unsigned int *)(a1 + 216);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 200), 80 * v4, 8);
  v8 = *(unsigned int *)(a1 + 184);
  if ( (_DWORD)v8 )
  {
    v9 = *(_QWORD **)(a1 + 168);
    v10 = &v9[10 * v8];
    do
    {
      if ( *v9 != -8192 && *v9 != -4096 )
      {
        v11 = v9[1];
        if ( (_QWORD *)v11 != v9 + 3 )
          _libc_free(v11);
      }
      v9 += 10;
    }
    while ( v10 != v9 );
    v8 = *(unsigned int *)(a1 + 184);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 168), 80 * v8, 8);
  v12 = *(_QWORD *)(a1 + 112);
  if ( v12 != a1 + 128 )
    _libc_free(v12);
  v13 = *(_QWORD *)(a1 + 64);
  if ( v13 != a1 + 80 )
    _libc_free(v13);
}
