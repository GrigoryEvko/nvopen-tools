// Function: sub_20E90A0
// Address: 0x20e90a0
//
__int64 __fastcall sub_20E90A0(__int64 a1)
{
  unsigned __int64 *v2; // rbx
  unsigned __int64 *v3; // r12
  unsigned __int64 v4; // rdi

  v2 = *(unsigned __int64 **)(a1 + 296);
  *(_QWORD *)a1 = &unk_4A00968;
  v3 = &v2[6 * *(unsigned int *)(a1 + 304)];
  if ( v2 != v3 )
  {
    do
    {
      v3 -= 6;
      if ( (unsigned __int64 *)*v3 != v3 + 2 )
        _libc_free(*v3);
    }
    while ( v2 != v3 );
    v3 = *(unsigned __int64 **)(a1 + 296);
  }
  if ( v3 != (unsigned __int64 *)(a1 + 312) )
    _libc_free((unsigned __int64)v3);
  v4 = *(_QWORD *)(a1 + 240);
  if ( v4 != a1 + 256 )
    _libc_free(v4);
  _libc_free(*(_QWORD *)(a1 + 208));
  _libc_free(*(_QWORD *)(a1 + 184));
  _libc_free(*(_QWORD *)(a1 + 160));
  *(_QWORD *)a1 = &unk_49EE078;
  sub_16366C0((_QWORD *)a1);
  return j_j___libc_free_0(a1, 504);
}
