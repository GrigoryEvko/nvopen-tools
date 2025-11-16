// Function: sub_3542290
// Address: 0x3542290
//
__int64 __fastcall sub_3542290(__int64 a1)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // r12
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  _QWORD *v10; // r14
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  unsigned __int64 *v15; // rbx
  unsigned __int64 *v16; // r12
  unsigned __int64 v17; // rdi

  v2 = *(_QWORD **)(a1 + 4088);
  v3 = *(_QWORD **)(a1 + 4080);
  *(_QWORD *)a1 = &unk_4A39258;
  if ( v2 != v3 )
  {
    do
    {
      if ( *v3 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v3 + 16LL))(*v3);
      ++v3;
    }
    while ( v2 != v3 );
    v3 = *(_QWORD **)(a1 + 4080);
  }
  if ( v3 )
    j_j___libc_free_0((unsigned __int64)v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 4056), 16LL * *(unsigned int *)(a1 + 4072), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 4024), 24LL * *(unsigned int *)(a1 + 4040), 8);
  v4 = *(_QWORD *)(a1 + 4000);
  if ( v4 != a1 + 4016 )
    _libc_free(v4);
  sub_C7D6A0(*(_QWORD *)(a1 + 3976), 8LL * *(unsigned int *)(a1 + 3992), 8);
  v5 = *(_QWORD *)(a1 + 3944);
  if ( v5 )
    j_j___libc_free_0(v5);
  v6 = *(_QWORD *)(a1 + 3872);
  if ( v6 != a1 + 3888 )
    _libc_free(v6);
  v7 = *(_QWORD *)(a1 + 3848);
  if ( v7 )
    j_j___libc_free_0(v7);
  v8 = *(_QWORD *)(a1 + 3824);
  if ( v8 )
    j_j___libc_free_0(v8);
  v9 = *(_QWORD *)(a1 + 3552);
  if ( v9 != a1 + 3568 )
    _libc_free(v9);
  v10 = *(_QWORD **)(a1 + 3464);
  if ( v10 )
  {
    v11 = v10[59];
    if ( (_QWORD *)v11 != v10 + 61 )
      _libc_free(v11);
    v12 = v10[41];
    if ( (_QWORD *)v12 != v10 + 43 )
      _libc_free(v12);
    v13 = v10[23];
    if ( (_QWORD *)v13 != v10 + 25 )
      _libc_free(v13);
    v14 = v10[5];
    if ( (_QWORD *)v14 != v10 + 7 )
      _libc_free(v14);
    v15 = (unsigned __int64 *)v10[3];
    v16 = (unsigned __int64 *)v10[2];
    if ( v15 != v16 )
    {
      do
      {
        v17 = v16[18];
        if ( (unsigned __int64 *)v17 != v16 + 20 )
          _libc_free(v17);
        if ( (unsigned __int64 *)*v16 != v16 + 2 )
          _libc_free(*v16);
        v16 += 36;
      }
      while ( v15 != v16 );
      v16 = (unsigned __int64 *)v10[2];
    }
    if ( v16 )
      j_j___libc_free_0((unsigned __int64)v16);
    j_j___libc_free_0((unsigned __int64)v10);
  }
  return sub_2EC45A0(a1);
}
