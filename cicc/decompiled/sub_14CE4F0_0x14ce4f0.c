// Function: sub_14CE4F0
// Address: 0x14ce4f0
//
__int64 __fastcall sub_14CE4F0(__int64 a1)
{
  void (*v1)(void); // rax

  v1 = *(void (**)(void))(*(_QWORD *)a1 + 128LL);
  if ( (char *)v1 == (char *)sub_14CE4E0 )
  {
    if ( byte_4F9D820 )
      sub_14CE140(a1);
    return 0;
  }
  else
  {
    v1();
    return 0;
  }
}
