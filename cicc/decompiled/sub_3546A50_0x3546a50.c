// Function: sub_3546A50
// Address: 0x3546a50
//
__int64 *__fastcall sub_3546A50(__int64 *a1)
{
  __int64 *result; // rax
  __int64 v2; // rdx
  __int64 *v3; // rcx
  __int64 v4; // rdx
  __int64 v5; // rcx

  result = a1;
  v2 = *a1 + 8;
  *a1 = v2;
  if ( v2 == a1[2] )
  {
    v3 = (__int64 *)(a1[3] + 8);
    a1[3] = (__int64)v3;
    v4 = *v3;
    v5 = *v3 + 512;
    a1[1] = v4;
    a1[2] = v5;
    *a1 = v4;
  }
  return result;
}
