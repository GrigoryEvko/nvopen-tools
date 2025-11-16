// Function: sub_350A640
// Address: 0x350a640
//
void __fastcall sub_350A640(__int64 a1, int a2)
{
  __int64 v3; // rdi
  __int64 (*v4)(); // rax
  __int64 *v5; // rbx
  unsigned __int64 *v6; // r12
  unsigned __int64 v7; // r13
  unsigned __int64 v8; // rdi

  v3 = *(_QWORD *)(a1 + 56);
  if ( v3 )
  {
    v4 = *(__int64 (**)())(*(_QWORD *)v3 + 32LL);
    if ( v4 == sub_2F60BA0 || (unsigned __int8)v4() )
    {
      v5 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 152LL) + 8LL * (a2 & 0x7FFFFFFF));
      v6 = (unsigned __int64 *)*v5;
      if ( *v5 )
      {
        sub_2E0AFD0(*v5);
        v7 = v6[12];
        if ( v7 )
        {
          sub_3509AB0(*(_QWORD *)(v7 + 16));
          j_j___libc_free_0(v7);
        }
        v8 = v6[8];
        if ( (unsigned __int64 *)v8 != v6 + 10 )
          _libc_free(v8);
        if ( (unsigned __int64 *)*v6 != v6 + 2 )
          _libc_free(*v6);
        j_j___libc_free_0((unsigned __int64)v6);
      }
      *v5 = 0;
    }
  }
}
