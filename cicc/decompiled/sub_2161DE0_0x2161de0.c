// Function: sub_2161DE0
// Address: 0x2161de0
//
void __fastcall sub_2161DE0(__int64 a1)
{
  _QWORD *v1; // r14
  __int64 v3; // rdx
  unsigned __int64 v4; // rdi
  _QWORD *v5; // rbx
  _QWORD *v6; // r12

  v1 = (_QWORD *)(a1 + 56);
  *(_QWORD *)a1 = &unk_4A01B58;
  *(_QWORD *)(a1 + 56) = &unk_4A02228;
  v3 = *(unsigned int *)(a1 + 360);
  v4 = *(_QWORD *)(a1 + 352);
  if ( v3 )
  {
    v5 = (_QWORD *)v4;
    do
    {
      v6 = (_QWORD *)*v5;
      if ( *v5 )
      {
        if ( (_QWORD *)*v6 != v6 + 2 )
          j_j___libc_free_0(*v6, v6[2] + 1LL);
        j_j___libc_free_0(v6, 32);
        v4 = *(_QWORD *)(a1 + 352);
        v3 = *(unsigned int *)(a1 + 360);
      }
      ++v5;
    }
    while ( v5 != (_QWORD *)(v4 + 8 * v3) );
  }
  if ( v4 != a1 + 368 )
    _libc_free(v4);
  *(_QWORD *)(a1 + 56) = &unk_4A02068;
  sub_1F4A9C0(v1);
  *(_QWORD *)a1 = &unk_4A012A0;
  nullsub_759();
}
