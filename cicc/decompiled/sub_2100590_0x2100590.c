// Function: sub_2100590
// Address: 0x2100590
//
void __fastcall sub_2100590(__int64 a1, int a2)
{
  __int64 v3; // rdi
  __int64 (*v4)(); // rax
  __int64 v5; // r13
  __int64 v6; // rbx
  __int64 *v7; // rax
  unsigned __int64 *v8; // r12
  unsigned __int64 v9; // r14
  unsigned __int64 v10; // rdi

  v3 = *(_QWORD *)(a1 + 56);
  if ( v3 )
  {
    v4 = *(__int64 (**)())(*(_QWORD *)v3 + 32LL);
    if ( v4 == sub_1ED7D00 || (unsigned __int8)v4() )
    {
      v5 = *(_QWORD *)(a1 + 32);
      v6 = 8LL * (a2 & 0x7FFFFFFF);
      v7 = (__int64 *)(v6 + *(_QWORD *)(v5 + 400));
      v8 = (unsigned __int64 *)*v7;
      if ( *v7 )
      {
        sub_1DB4CE0(*v7);
        v9 = v8[12];
        if ( v9 )
        {
          sub_20FF8D0(*(_QWORD *)(v9 + 16));
          j_j___libc_free_0(v9, 48);
        }
        v10 = v8[8];
        if ( (unsigned __int64 *)v10 != v8 + 10 )
          _libc_free(v10);
        if ( (unsigned __int64 *)*v8 != v8 + 2 )
          _libc_free(*v8);
        j_j___libc_free_0(v8, 120);
        v7 = (__int64 *)(v6 + *(_QWORD *)(v5 + 400));
      }
      *v7 = 0;
    }
  }
}
