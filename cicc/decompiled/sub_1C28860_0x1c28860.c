// Function: sub_1C28860
// Address: 0x1c28860
//
void *__fastcall sub_1C28860(_QWORD *a1)
{
  _QWORD *v2; // rdi

  *a1 = &unk_49F7658;
  v2 = (_QWORD *)a1[21];
  if ( v2 != a1 + 23 )
    j_j___libc_free_0(v2, a1[23] + 1LL);
  *a1 = &unk_49EE078;
  return sub_16366C0(a1);
}
