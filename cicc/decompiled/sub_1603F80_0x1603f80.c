// Function: sub_1603F80
// Address: 0x1603f80
//
void __fastcall sub_1603F80(__int64 a1)
{
  void (*v1)(void); // rax

  if ( a1 )
  {
    v1 = *(void (**)(void))(*(_QWORD *)a1 + 8LL);
    if ( (char *)v1 == (char *)sub_1603E10 )
      j_j___libc_free_0(a1, 16);
    else
      v1();
  }
}
