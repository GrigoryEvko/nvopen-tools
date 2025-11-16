// Function: sub_1C077E0
// Address: 0x1c077e0
//
void *__fastcall sub_1C077E0(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = &unk_49F74A0;
  v2 = a1[20];
  if ( v2 )
    j_j___libc_free_0(v2, 32);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
