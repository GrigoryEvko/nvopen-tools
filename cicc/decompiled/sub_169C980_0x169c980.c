// Function: sub_169C980
// Address: 0x169c980
//
__int64 __fastcall sub_169C980(void **a1, unsigned __int8 a2)
{
  void **v2; // r12
  void **v3; // rdi
  void *v4; // rbx

  do
  {
    v2 = a1;
    v3 = (void **)((char *)a1[1] + 8);
    v4 = sub_16982C0();
    if ( *v3 == v4 )
      sub_169C980(v3, a2);
    else
      sub_169B620((__int64)v3, a2);
    a2 = 0;
    a1 = (void **)((char *)v2[1] + 40);
  }
  while ( *a1 == v4 );
  return sub_169B620((__int64)a1, 0);
}
