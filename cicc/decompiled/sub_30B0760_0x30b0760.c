// Function: sub_30B0760
// Address: 0x30b0760
//
void __fastcall sub_30B0760(__int64 a1)
{
  __int64 *v2; // r13
  __int64 v3; // r12
  unsigned __int64 *v4; // rax
  unsigned __int64 *v5; // rbx
  unsigned __int64 *v6; // r15
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  __int64 *v9; // [rsp+8h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 96);
  *(_QWORD *)a1 = &unk_4A32470;
  v9 = &v2[*(unsigned int *)(a1 + 104)];
  while ( v9 != v2 )
  {
    v3 = *v2;
    v4 = *(unsigned __int64 **)(*v2 + 40);
    v5 = &v4[*(unsigned int *)(*v2 + 48)];
    if ( v5 != v4 )
    {
      v6 = *(unsigned __int64 **)(*v2 + 40);
      do
      {
        if ( *v6 )
          j_j___libc_free_0(*v6);
        ++v6;
      }
      while ( v5 != v6 );
    }
    ++v2;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v3 + 8LL))(v3);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 200), 16LL * *(unsigned int *)(a1 + 216), 8);
  v7 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)a1 = &unk_4A32388;
  if ( v7 != a1 + 24 )
    j_j___libc_free_0(v7);
  v8 = *(_QWORD *)(a1 + 96);
  if ( v8 != a1 + 112 )
    _libc_free(v8);
}
