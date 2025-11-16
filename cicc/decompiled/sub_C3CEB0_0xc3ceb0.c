// Function: sub_C3CEB0
// Address: 0xc3ceb0
//
__int64 __fastcall sub_C3CEB0(void **a1, unsigned __int8 a2)
{
  void **v2; // r12
  void **v3; // rdi
  void *v4; // rbx

  do
  {
    v2 = a1;
    v3 = (void **)a1[1];
    v4 = sub_C33340();
    if ( *v3 == v4 )
      sub_C3CEB0(v3, a2);
    else
      sub_C37310((__int64)v3, a2);
    a2 = 0;
    a1 = (void **)((char *)v2[1] + 24);
  }
  while ( *a1 == v4 );
  return sub_C37310((__int64)a1, 0);
}
