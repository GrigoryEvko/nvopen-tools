// Function: sub_2D2B870
// Address: 0x2d2b870
//
__int64 __fastcall sub_2D2B870(_QWORD *a1, __int64 *a2)
{
  __int64 *v2; // rax
  __int64 result; // rax

  v2 = sub_2D2B810(a1, (unsigned __int64)*a2 % a1[1], a2, *a2);
  if ( !v2 )
    return 0;
  result = *v2;
  if ( !result )
    return 0;
  return result;
}
