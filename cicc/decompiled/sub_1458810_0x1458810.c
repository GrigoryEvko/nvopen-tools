// Function: sub_1458810
// Address: 0x1458810
//
void __fastcall sub_1458810(__int64 a1)
{
  unsigned __int64 v1; // r13
  unsigned __int64 v2; // r12
  __int64 v3; // r15
  __int64 v4; // rax
  _QWORD *v5; // rbx
  _QWORD *v6; // r14
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi

  v1 = *(_QWORD *)a1;
  v2 = *(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v2 )
  {
    do
    {
      v3 = *(_QWORD *)(v2 - 8);
      v2 -= 24LL;
      if ( v3 )
      {
        *(_QWORD *)v3 = &unk_49EC708;
        v4 = *(unsigned int *)(v3 + 208);
        if ( (_DWORD)v4 )
        {
          v5 = *(_QWORD **)(v3 + 192);
          v6 = &v5[7 * v4];
          do
          {
            if ( *v5 != -8 && *v5 != -16 )
            {
              v7 = v5[1];
              if ( (_QWORD *)v7 != v5 + 3 )
                _libc_free(v7);
            }
            v5 += 7;
          }
          while ( v6 != v5 );
        }
        j___libc_free_0(*(_QWORD *)(v3 + 192));
        v8 = *(_QWORD *)(v3 + 40);
        if ( v8 != v3 + 56 )
          _libc_free(v8);
        j_j___libc_free_0(v3, 216);
      }
    }
    while ( v1 != v2 );
    v2 = *(_QWORD *)a1;
  }
  if ( v2 != a1 + 16 )
    _libc_free(v2);
}
