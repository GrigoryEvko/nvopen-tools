// Function: sub_8CBAA0
// Address: 0x8cbaa0
//
void __fastcall sub_8CBAA0(__int64 a1)
{
  __int64 **v1; // rax

  v1 = *(__int64 ***)(a1 + 32);
  if ( v1 && qword_4F074B0 == qword_4F60258 )
  {
    if ( (*((_BYTE *)*v1 + 195) & 2) != 0 && (*(_BYTE *)(a1 + 195) & 2) == 0 )
      sub_8C6700(*v1, (unsigned int *)(a1 + 64), 0x42Au, 0x425u);
    sub_8CB6C0(0xBu, a1);
  }
}
