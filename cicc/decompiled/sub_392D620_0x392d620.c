// Function: sub_392D620
// Address: 0x392d620
//
void __fastcall sub_392D620(unsigned __int64 a1)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // r13
  _QWORD *v4; // rbx
  _QWORD *v5; // r13
  unsigned __int64 v6; // rdi
  __int64 v7; // rdi

  *(_QWORD *)a1 = &unk_4A3EE20;
  v2 = *(_QWORD *)(a1 + 88);
  if ( v2 )
    j_j___libc_free_0(v2);
  j___libc_free_0(*(_QWORD *)(a1 + 56));
  v3 = *(unsigned int *)(a1 + 40);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD **)(a1 + 24);
    v5 = &v4[4 * v3];
    do
    {
      if ( *v4 != -16 && *v4 != -8 )
      {
        v6 = v4[1];
        if ( v6 )
          j_j___libc_free_0(v6);
      }
      v4 += 4;
    }
    while ( v5 != v4 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 24));
  v7 = *(_QWORD *)(a1 + 8);
  if ( v7 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 8LL))(v7);
  nullsub_1935();
  j_j___libc_free_0(a1);
}
