// Function: sub_8CB9C0
// Address: 0x8cb9c0
//
void __fastcall sub_8CB9C0(__int64 a1)
{
  __int64 **v1; // rax

  v1 = *(__int64 ***)(a1 + 32);
  if ( v1 && qword_4F074B0 == qword_4F60258 )
  {
    if ( *((char *)*v1 + 170) < 0 )
      sub_8C6700(*v1, (unsigned int *)(a1 + 64), 0x42Au, 0x425u);
    sub_8CB6C0(7u, a1);
  }
}
