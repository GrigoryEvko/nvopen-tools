// Function: sub_22527D0
// Address: 0x22527d0
//
void __fastcall sub_22527D0(__int64 a1)
{
  unsigned __int64 v1; // rdi

  v1 = a1 - 128;
  if ( v1 <= qword_4FD6AD0 || v1 >= qword_4FD6AD8 + qword_4FD6AD0 )
    _libc_free(v1);
  else
    sub_2252580(v1);
}
