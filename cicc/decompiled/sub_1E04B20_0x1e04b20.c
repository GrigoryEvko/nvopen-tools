// Function: sub_1E04B20
// Address: 0x1e04b20
//
void __fastcall sub_1E04B20(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // r12
  _QWORD *v3; // rbx
  _QWORD *v4; // r12
  __int64 v5; // r14
  __int64 v6; // rdi

  v1 = *(_QWORD *)(a1 + 1312);
  *(_DWORD *)(a1 + 240) = 0;
  *(_QWORD *)(a1 + 1312) = 0;
  if ( v1 )
  {
    v2 = *(unsigned int *)(v1 + 48);
    if ( (_DWORD)v2 )
    {
      v3 = *(_QWORD **)(v1 + 32);
      v4 = &v3[2 * v2];
      do
      {
        if ( *v3 != -16 && *v3 != -8 )
        {
          v5 = v3[1];
          if ( v5 )
          {
            v6 = *(_QWORD *)(v5 + 24);
            if ( v6 )
              j_j___libc_free_0(v6, *(_QWORD *)(v5 + 40) - v6);
            j_j___libc_free_0(v5, 56);
          }
        }
        v3 += 2;
      }
      while ( v4 != v3 );
    }
    j___libc_free_0(*(_QWORD *)(v1 + 32));
    if ( *(_QWORD *)v1 != v1 + 16 )
      _libc_free(*(_QWORD *)v1);
    j_j___libc_free_0(v1, 80);
  }
}
