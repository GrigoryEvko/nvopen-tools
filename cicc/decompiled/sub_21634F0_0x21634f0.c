// Function: sub_21634F0
// Address: 0x21634f0
//
__int64 __fastcall sub_21634F0(__int64 a1)
{
  __int64 v2; // rdx
  unsigned __int64 v3; // rdi
  _QWORD *v4; // rbx
  _QWORD *v5; // r12

  *(_QWORD *)a1 = &unk_4A02228;
  v2 = *(unsigned int *)(a1 + 304);
  v3 = *(_QWORD *)(a1 + 296);
  if ( v2 )
  {
    v4 = (_QWORD *)v3;
    do
    {
      v5 = (_QWORD *)*v4;
      if ( *v4 )
      {
        if ( (_QWORD *)*v5 != v5 + 2 )
          j_j___libc_free_0(*v5, v5[2] + 1LL);
        j_j___libc_free_0(v5, 32);
        v3 = *(_QWORD *)(a1 + 296);
        v2 = *(unsigned int *)(a1 + 304);
      }
      ++v4;
    }
    while ( v4 != (_QWORD *)(v3 + 8 * v2) );
  }
  if ( v3 != a1 + 312 )
    _libc_free(v3);
  *(_QWORD *)a1 = &unk_4A02068;
  return sub_1F4A9C0((_QWORD *)a1);
}
