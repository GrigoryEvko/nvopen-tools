// Function: sub_1422230
// Address: 0x1422230
//
__int64 __fastcall sub_1422230(_QWORD *a1)
{
  __int64 v1; // r13

  v1 = a1[20];
  *a1 = &unk_49EB2E8;
  if ( v1 )
  {
    sub_1421ED0(v1);
    j_j___libc_free_0(v1, 344);
  }
  *a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 168);
}
