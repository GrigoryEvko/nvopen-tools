// Function: sub_70C920
// Address: 0x70c920
//
__int64 __fastcall sub_70C920(unsigned __int8 *a1)
{
  unsigned __int8 *v1; // rcx
  __int64 result; // rax
  int v3; // edx

  v1 = a1 + 16;
  LODWORD(result) = 0;
  do
  {
    v3 = *a1++;
    result = (unsigned int)(v3 + result);
  }
  while ( v1 != a1 );
  return result;
}
