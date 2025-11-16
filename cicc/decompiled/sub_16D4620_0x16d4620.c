// Function: sub_16D4620
// Address: 0x16d4620
//
__int64 __fastcall sub_16D4620(_QWORD *a1)
{
  void (***v2)(void); // rdi
  void (*v3)(void); // rdx

  *a1 = &unk_49EF4E8;
  v2 = (void (***)(void))a1[1];
  if ( v2 )
  {
    v3 = **v2;
    if ( (char *)v3 == (char *)sub_16D4120 )
      (*v2)[2]();
    else
      v3();
  }
  return j_j___libc_free_0(a1, 32);
}
