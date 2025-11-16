// Function: sub_2305CF0
// Address: 0x2305cf0
//
void __fastcall sub_2305CF0(unsigned __int64 a1)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // rax
  _QWORD *v4; // rbx
  _QWORD *v5; // r13
  unsigned __int64 v6; // rdi

  *(_QWORD *)a1 = &unk_4A0ACC8;
  v2 = *(_QWORD *)(a1 + 48);
  if ( v2 )
    j_j___libc_free_0(v2);
  v3 = *(unsigned int *)(a1 + 40);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD **)(a1 + 24);
    v5 = &v4[11 * v3];
    do
    {
      if ( *v4 != -8192 && *v4 != -4096 )
      {
        v6 = v4[2];
        if ( (_QWORD *)v6 != v4 + 4 )
          _libc_free(v6);
      }
      v4 += 11;
    }
    while ( v5 != v4 );
    v3 = *(unsigned int *)(a1 + 40);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 24), 88 * v3, 8);
  j_j___libc_free_0(a1);
}
