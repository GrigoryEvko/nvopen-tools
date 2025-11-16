// Function: sub_214B330
// Address: 0x214b330
//
__int64 __fastcall sub_214B330(__int64 a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r12

  v1 = *(_QWORD **)(a1 + 8);
  v2 = &v1[4 * *(unsigned int *)(a1 + 16)];
  *(_QWORD *)a1 = &unk_4A016E8;
  if ( v1 != v2 )
  {
    do
    {
      v2 -= 4;
      if ( (_QWORD *)*v2 != v2 + 2 )
        j_j___libc_free_0(*v2, v2[2] + 1LL);
    }
    while ( v1 != v2 );
    v2 = *(_QWORD **)(a1 + 8);
  }
  if ( v2 != (_QWORD *)(a1 + 24) )
    _libc_free((unsigned __int64)v2);
  nullsub_702();
  return j_j___libc_free_0(a1, 280);
}
