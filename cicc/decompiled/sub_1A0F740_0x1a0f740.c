// Function: sub_1A0F740
// Address: 0x1a0f740
//
void __fastcall sub_1A0F740(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r13
  unsigned __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rdi

  j___libc_free_0(*(_QWORD *)(a1 + 2408));
  v2 = *(_QWORD *)(a1 + 1872);
  if ( v2 != a1 + 1888 )
    _libc_free(v2);
  v3 = *(_QWORD *)(a1 + 1344);
  if ( v3 != a1 + 1360 )
    _libc_free(v3);
  v4 = *(_QWORD *)(a1 + 816);
  if ( v4 != a1 + 832 )
    _libc_free(v4);
  v5 = *(_QWORD *)(a1 + 664);
  if ( v5 != *(_QWORD *)(a1 + 656) )
    _libc_free(v5);
  v6 = *(_QWORD *)(a1 + 496);
  if ( v6 != *(_QWORD *)(a1 + 488) )
    _libc_free(v6);
  v7 = *(_QWORD *)(a1 + 328);
  if ( v7 != *(_QWORD *)(a1 + 320) )
    _libc_free(v7);
  j___libc_free_0(*(_QWORD *)(a1 + 288));
  j___libc_free_0(*(_QWORD *)(a1 + 256));
  j___libc_free_0(*(_QWORD *)(a1 + 224));
  j___libc_free_0(*(_QWORD *)(a1 + 192));
  v8 = *(unsigned int *)(a1 + 176);
  if ( (_DWORD)v8 )
  {
    v9 = *(_QWORD *)(a1 + 160);
    v10 = v9 + 48 * v8;
    do
    {
      if ( *(_QWORD *)v9 != -16 && *(_QWORD *)v9 != -8 && *(_DWORD *)(v9 + 8) == 3 )
      {
        if ( *(_DWORD *)(v9 + 40) > 0x40u )
        {
          v12 = *(_QWORD *)(v9 + 32);
          if ( v12 )
            j_j___libc_free_0_0(v12);
        }
        if ( *(_DWORD *)(v9 + 24) > 0x40u )
        {
          v13 = *(_QWORD *)(v9 + 16);
          if ( v13 )
            j_j___libc_free_0_0(v13);
        }
      }
      v9 += 48;
    }
    while ( v10 != v9 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 160));
  j___libc_free_0(*(_QWORD *)(a1 + 128));
  v11 = *(_QWORD *)(a1 + 32);
  if ( v11 != *(_QWORD *)(a1 + 24) )
    _libc_free(v11);
}
