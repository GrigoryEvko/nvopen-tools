// Function: sub_131A8A0
// Address: 0x131a8a0
//
int __fastcall sub_131A8A0(__int64 a1)
{
  unsigned int v1; // r13d
  __int64 v2; // rax

  if ( qword_5260D48[0] )
  {
    v1 = 0;
    v2 = 0;
    do
    {
      sub_130B050(a1, qword_5260DD8 + 208 * v2 + 56);
      v2 = ++v1;
    }
    while ( (unsigned __int64)v1 < qword_5260D48[0] );
  }
  return sub_130B050(a1, (__int64)&unk_5260D60);
}
