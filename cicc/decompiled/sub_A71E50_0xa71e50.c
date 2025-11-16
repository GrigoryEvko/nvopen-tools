// Function: sub_A71E50
// Address: 0xa71e50
//
unsigned __int64 __fastcall sub_A71E50(__int64 *a1)
{
  __int64 v1; // rax
  unsigned int v2; // edx

  v1 = sub_A71B70(*a1);
  v2 = v1;
  if ( (_DWORD)v1 == -1 )
    v2 = 0;
  return __PAIR64__(v2, HIDWORD(v1));
}
