// Function: sub_F53C40
// Address: 0xf53c40
//
__int64 __fastcall sub_F53C40(int a1)
{
  __int64 v1; // rdi
  __int64 result; // rax

  v1 = (unsigned int)(a1 - 32);
  result = 0;
  if ( (unsigned int)v1 <= 9 )
    return *(_QWORD *)&asc_3F8A840[8 * v1];
  return result;
}
