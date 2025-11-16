// Function: sub_1436910
// Address: 0x1436910
//
__int64 __fastcall sub_1436910(__int64 a1)
{
  __int64 v2; // rax
  _QWORD *v3; // rbx
  _QWORD *v4; // r13
  unsigned __int64 v5; // rdi

  *(_QWORD *)a1 = off_49EB690;
  v2 = *(unsigned int *)(a1 + 32);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD **)(a1 + 16);
    v4 = &v3[7 * v2];
    do
    {
      if ( *v3 != -16 && *v3 != -8 )
      {
        v5 = v3[1];
        if ( (_QWORD *)v5 != v3 + 3 )
          _libc_free(v5);
      }
      v3 += 7;
    }
    while ( v4 != v3 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 16));
  return nullsub_544(a1);
}
