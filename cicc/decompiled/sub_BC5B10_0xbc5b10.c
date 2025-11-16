// Function: sub_BC5B10
// Address: 0xbc5b10
//
void __fastcall sub_BC5B10(_QWORD *a1)
{
  if ( (a1[4] & 1) == 0 && (_QWORD *)*a1 != a1 + 2 )
    j_j___libc_free_0(*a1, a1[2] + 1LL);
}
