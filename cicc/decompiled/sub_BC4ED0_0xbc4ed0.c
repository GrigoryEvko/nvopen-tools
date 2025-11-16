// Function: sub_BC4ED0
// Address: 0xbc4ed0
//
void __fastcall sub_BC4ED0(char *a1, __int64 a2)
{
  if ( *(char **)a1 != a1 + 16 )
    _libc_free(*(_QWORD *)a1, a2);
}
