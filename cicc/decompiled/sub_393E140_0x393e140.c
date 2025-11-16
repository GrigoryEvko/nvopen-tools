// Function: sub_393E140
// Address: 0x393e140
//
void __fastcall sub_393E140(_QWORD *a1)
{
  _QWORD *v1; // r14
  unsigned __int64 v2; // r13
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // r12
  __int64 v5; // rdi
  unsigned __int64 v6; // rdi

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = (unsigned __int64)v1;
      sub_393E140(v1[3]);
      v3 = v1[20];
      v1 = (_QWORD *)v1[2];
      while ( v3 )
      {
        v4 = v3;
        sub_393DEF0(*(_QWORD **)(v3 + 24));
        v5 = *(_QWORD *)(v3 + 56);
        v3 = *(_QWORD *)(v3 + 16);
        sub_393E140(v5);
        j_j___libc_free_0(v4);
      }
      sub_393DB20(*(_QWORD *)(v2 + 112));
      v6 = *(_QWORD *)(v2 + 32);
      if ( v6 != v2 + 48 )
        j_j___libc_free_0(v6);
      j_j___libc_free_0(v2);
    }
    while ( v1 );
  }
}
