// Function: sub_BB8B30
// Address: 0xbb8b30
//
__int64 __fastcall sub_BB8B30(__int64 a1, __int64 a2)
{
  void (__fastcall *v3)(__int64, __int64, __int64); // rax
  __int64 v4; // rdi
  _QWORD *v5; // rbx
  _QWORD *v6; // r12
  _QWORD *v7; // rdi
  _QWORD *v8; // rbx
  _QWORD *v9; // r12
  __int64 v10; // rdi

  *(_QWORD *)a1 = &unk_49DAD08;
  v3 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 240);
  if ( v3 )
  {
    a2 = a1 + 224;
    v3(a2, a2, 3);
  }
  v4 = *(_QWORD *)(a1 + 192);
  if ( v4 )
  {
    a2 = *(_QWORD *)(a1 + 208) - v4;
    j_j___libc_free_0(v4, a2);
  }
  v5 = *(_QWORD **)(a1 + 168);
  v6 = *(_QWORD **)(a1 + 160);
  if ( v5 != v6 )
  {
    do
    {
      v7 = (_QWORD *)v6[1];
      *v6 = &unk_49DACE8;
      if ( v7 != v6 + 3 )
      {
        a2 = v6[3] + 1LL;
        j_j___libc_free_0(v7, a2);
      }
      v6 += 6;
    }
    while ( v5 != v6 );
    v6 = *(_QWORD **)(a1 + 160);
  }
  if ( v6 )
  {
    a2 = *(_QWORD *)(a1 + 176) - (_QWORD)v6;
    j_j___libc_free_0(v6, a2);
  }
  v8 = *(_QWORD **)(a1 + 144);
  v9 = *(_QWORD **)(a1 + 136);
  if ( v8 != v9 )
  {
    do
    {
      if ( (_QWORD *)*v9 != v9 + 2 )
      {
        a2 = v9[2] + 1LL;
        j_j___libc_free_0(*v9, a2);
      }
      v9 += 4;
    }
    while ( v8 != v9 );
    v9 = *(_QWORD **)(a1 + 136);
  }
  if ( v9 )
  {
    a2 = *(_QWORD *)(a1 + 152) - (_QWORD)v9;
    j_j___libc_free_0(v9, a2);
  }
  if ( !*(_BYTE *)(a1 + 124) )
    _libc_free(*(_QWORD *)(a1 + 104), a2);
  v10 = *(_QWORD *)(a1 + 72);
  if ( v10 != a1 + 88 )
    _libc_free(v10, a2);
  return j_j___libc_free_0(a1, 256);
}
