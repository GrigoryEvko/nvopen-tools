// Function: sub_2DB8C10
// Address: 0x2db8c10
//
__int64 __fastcall sub_2DB8C10(_QWORD *a1)
{
  unsigned __int64 v2; // r14
  unsigned __int64 *v3; // rbx
  unsigned __int64 *v4; // r12
  unsigned __int64 v5; // rdi

  v2 = a1[25];
  *a1 = &unk_4A275D8;
  if ( v2 )
  {
    v3 = *(unsigned __int64 **)(v2 + 64);
    v4 = &v3[6 * *(unsigned int *)(v2 + 72)];
    if ( v3 != v4 )
    {
      do
      {
        v4 -= 6;
        if ( (unsigned __int64 *)*v4 != v4 + 2 )
          _libc_free(*v4);
      }
      while ( v3 != v4 );
      v4 = *(unsigned __int64 **)(v2 + 64);
    }
    if ( v4 != (unsigned __int64 *)(v2 + 80) )
      _libc_free((unsigned __int64)v4);
    v5 = *(_QWORD *)(v2 + 8);
    if ( v5 != v2 + 24 )
      _libc_free(v5);
    j_j___libc_free_0(v2);
  }
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
