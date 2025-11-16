// Function: sub_214BDC0
// Address: 0x214bdc0
//
__int64 __fastcall sub_214BDC0(__int64 a1)
{
  __int64 *v2; // r12
  __int64 *v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // rbx
  __int64 v6; // r12
  __int64 v7; // rdi
  __int64 v8; // rax
  _QWORD *v9; // rbx
  _QWORD *v10; // r12
  __int64 v11; // rdi

  v2 = *(__int64 **)(a1 + 904);
  *(_QWORD *)a1 = off_4A01708;
  if ( v2 )
  {
    v3 = (__int64 *)v2[8];
    if ( v3 != v2 + 10 )
      j_j___libc_free_0(v3, v2[10] + 1);
    v4 = v2[7];
    if ( v4 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v4 + 8LL))(v4);
    sub_214B5A0(v2[2]);
    j_j___libc_free_0(v2, 96);
  }
  v5 = *(_QWORD *)(a1 + 864);
  while ( v5 )
  {
    v6 = v5;
    sub_214B820(*(_QWORD **)(v5 + 24));
    v7 = *(_QWORD *)(v5 + 40);
    v5 = *(_QWORD *)(v5 + 16);
    if ( v7 )
      j_j___libc_free_0(v7, *(_QWORD *)(v6 + 56) - v7);
    j_j___libc_free_0(v6, 64);
  }
  v8 = *(unsigned int *)(a1 + 832);
  if ( (_DWORD)v8 )
  {
    v9 = *(_QWORD **)(a1 + 816);
    v10 = &v9[5 * v8];
    do
    {
      if ( *v9 != -16 && *v9 != -8 )
        j___libc_free_0(v9[2]);
      v9 += 5;
    }
    while ( v10 != v9 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 816));
  v11 = *(_QWORD *)(a1 + 752);
  if ( v11 != a1 + 768 )
    j_j___libc_free_0(v11, *(_QWORD *)(a1 + 768) + 1LL);
  return sub_396C410(a1);
}
