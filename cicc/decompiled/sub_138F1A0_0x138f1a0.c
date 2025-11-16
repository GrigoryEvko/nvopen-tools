// Function: sub_138F1A0
// Address: 0x138f1a0
//
__int64 __fastcall sub_138F1A0(__int64 a1)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // r12
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // rbx
  __int64 v7; // r12
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // rdi

  v2 = *(_QWORD **)(a1 + 48);
  while ( v2 )
  {
    v3 = v2;
    v2 = (_QWORD *)*v2;
    v4 = v3[4];
    v3[1] = &unk_49EE2B0;
    if ( v4 != -8 && v4 != 0 && v4 != -16 )
      sub_1649B30(v3 + 2);
    j_j___libc_free_0(v3, 48);
  }
  v5 = *(unsigned int *)(a1 + 40);
  if ( (_DWORD)v5 )
  {
    v6 = *(_QWORD *)(a1 + 24);
    v7 = v6 + 424 * v5;
    do
    {
      while ( *(_QWORD *)v6 == -8 || *(_QWORD *)v6 == -16 || !*(_BYTE *)(v6 + 416) )
      {
        v6 += 424;
        if ( v7 == v6 )
          return j___libc_free_0(*(_QWORD *)(a1 + 24));
      }
      v8 = *(_QWORD *)(v6 + 272);
      if ( v8 != v6 + 288 )
        _libc_free(v8);
      v9 = *(_QWORD *)(v6 + 64);
      if ( v9 != v6 + 80 )
        _libc_free(v9);
      v10 = *(_QWORD *)(v6 + 40);
      if ( v10 )
        j_j___libc_free_0(v10, *(_QWORD *)(v6 + 56) - v10);
      v11 = *(_QWORD *)(v6 + 16);
      v6 += 424;
      j___libc_free_0(v11);
    }
    while ( v7 != v6 );
  }
  return j___libc_free_0(*(_QWORD *)(a1 + 24));
}
