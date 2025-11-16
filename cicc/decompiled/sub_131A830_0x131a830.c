// Function: sub_131A830
// Address: 0x131a830
//
void __fastcall sub_131A830(__int64 a1)
{
  unsigned int v1; // r14d
  __int64 v2; // rax

  if ( qword_5260D48[0] )
  {
    v1 = 0;
    v2 = 0;
    do
    {
      sub_130B000(a1, qword_5260DD8 + 208 * v2 + 56);
      v2 = ++v1;
    }
    while ( (unsigned __int64)v1 < qword_5260D48[0] );
  }
}
