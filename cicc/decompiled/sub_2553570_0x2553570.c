// Function: sub_2553570
// Address: 0x2553570
//
void __fastcall sub_2553570(unsigned __int64 *a1)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rdi

  v2 = a1[8];
  while ( v2 )
  {
    sub_253B2D0(*(_QWORD *)(v2 + 24));
    v3 = v2;
    v2 = *(_QWORD *)(v2 + 16);
    j_j___libc_free_0(v3);
  }
  if ( (unsigned __int64 *)*a1 != a1 + 2 )
    _libc_free(*a1);
}
