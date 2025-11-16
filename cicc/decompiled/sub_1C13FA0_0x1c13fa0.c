// Function: sub_1C13FA0
// Address: 0x1c13fa0
//
__int64 __fastcall sub_1C13FA0(__int64 a1)
{
  __int64 (*v1)(void); // rax

  v1 = *(__int64 (**)(void))(*(_QWORD *)a1 + 56LL);
  if ( (char *)v1 != (char *)sub_1C13F70 )
    return v1();
  sub_1C13E30(a1);
  return j_j___libc_free_0(a1, 152);
}
