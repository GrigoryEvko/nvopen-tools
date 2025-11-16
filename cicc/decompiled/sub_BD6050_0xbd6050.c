// Function: sub_BD6050
// Address: 0xbd6050
//
__int64 __fastcall sub_BD6050(unsigned __int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 *v2; // rdx
  __int64 result; // rax

  a1[1] = *(_QWORD *)a2;
  *(_QWORD *)a2 = a1;
  v2 = (unsigned __int64 *)a1[1];
  result = *a1 & 7;
  *a1 = result | a2;
  if ( v2 )
  {
    result = *v2 & 7;
    *v2 = result | (unsigned __int64)(a1 + 1);
  }
  return result;
}
