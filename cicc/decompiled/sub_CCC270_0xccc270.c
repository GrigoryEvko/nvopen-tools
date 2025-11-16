// Function: sub_CCC270
// Address: 0xccc270
//
__int64 __fastcall sub_CCC270(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 (*v4)(void); // rax

  v4 = *(__int64 (**)(void))(*(_QWORD *)a1 + 56LL);
  if ( (char *)v4 != (char *)sub_CCC240 )
    return v4();
  sub_CCC0B0(a1, a2, (__int64)sub_CCC240, a4);
  return j_j___libc_free_0(a1, 152);
}
