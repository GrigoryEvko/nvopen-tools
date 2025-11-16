// Function: sub_1E6C740
// Address: 0x1e6c740
//
__int64 __fastcall sub_1E6C740(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rdi

  v1 = a1 - 8;
  v2 = a1 + 32;
  unk_4FC7860 = 0;
  if ( *(_QWORD *)(v2 - 16) != v2 )
    _libc_free(*(_QWORD *)(v2 - 16));
  return j_j___libc_free_0(v1, 488);
}
