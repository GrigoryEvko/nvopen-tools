// Function: sub_130F150
// Address: 0x130f150
//
__int64 __fastcall sub_130F150(unsigned int *a1)
{
  const char *v1; // r14
  __int64 result; // rax
  int v3; // r13d
  int i; // ebx

  v1 = "\t";
  result = *a1;
  v3 = a1[6];
  if ( (_DWORD)result )
  {
    v3 *= 2;
    v1 = " ";
  }
  if ( v3 > 0 )
  {
    for ( i = 0; i != v3; ++i )
      result = sub_130F0B0((__int64)a1, "%s", v1);
  }
  return result;
}
