// Function: sub_B96F80
// Address: 0xb96f80
//
__int64 __fastcall sub_B96F80(__int64 *a1)
{
  __int64 *v1; // rbx
  __int64 result; // rax

  v1 = a1;
  do
  {
    if ( *v1 )
      result = sub_B96E90((__int64)v1, *v1, (unsigned __int64)a1 & 0xFFFFFFFFFFFFFFFCLL | 2);
    ++v1;
  }
  while ( v1 != a1 + 3 );
  return result;
}
