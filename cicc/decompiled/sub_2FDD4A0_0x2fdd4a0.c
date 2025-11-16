// Function: sub_2FDD4A0
// Address: 0x2fdd4a0
//
void __fastcall sub_2FDD4A0(_QWORD *a1)
{
  unsigned __int64 v1; // rdi
  void (*v2)(void); // rax

  *a1 = &unk_4A2C618;
  v1 = a1[7];
  if ( v1 )
  {
    v2 = *(void (**)(void))(*(_QWORD *)v1 + 8LL);
    if ( (char *)v2 == (char *)sub_2EAAD20 )
      j_j___libc_free_0(v1);
    else
      v2();
  }
}
