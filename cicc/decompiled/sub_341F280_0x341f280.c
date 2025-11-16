// Function: sub_341F280
// Address: 0x341f280
//
void __fastcall sub_341F280(__int64 a1)
{
  unsigned __int64 v1; // r12
  __int64 v2; // rdi

  v1 = a1 - 8;
  v2 = a1 + 32;
  unk_5039AC0 = 0;
  if ( *(_QWORD *)(v2 - 16) != v2 )
    _libc_free(*(_QWORD *)(v2 - 16));
  j_j___libc_free_0(v1);
}
