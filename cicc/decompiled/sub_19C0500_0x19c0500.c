// Function: sub_19C0500
// Address: 0x19c0500
//
void __fastcall sub_19C0500(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // r14
  __int64 v3; // rax
  _QWORD *v4; // rbx
  _QWORD *v5; // r12
  unsigned __int64 v6; // rdi

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      sub_19C0500(*(_QWORD *)(v1 + 24));
      v3 = *(unsigned int *)(v1 + 80);
      v1 = *(_QWORD *)(v1 + 16);
      if ( (_DWORD)v3 )
      {
        v4 = *(_QWORD **)(v2 + 64);
        v5 = &v4[14 * v3];
        do
        {
          if ( *v4 != -16 && *v4 != -8 )
          {
            v6 = v4[3];
            if ( v6 != v4[2] )
              _libc_free(v6);
          }
          v4 += 14;
        }
        while ( v5 != v4 );
      }
      j___libc_free_0(*(_QWORD *)(v2 + 64));
      j_j___libc_free_0(v2, 88);
    }
    while ( v1 );
  }
}
