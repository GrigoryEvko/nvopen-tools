// Function: sub_38E30D0
// Address: 0x38e30d0
//
__int64 __fastcall sub_38E30D0(_QWORD *a1, _QWORD *a2)
{
  unsigned __int64 v2; // rdx
  __int64 result; // rax

  v2 = a2[1];
  result = 0xFFFFFFFFLL;
  if ( a1[1] >= v2 )
  {
    result = 1;
    if ( a1[1] <= v2 )
      return 2 * (unsigned int)(byte_452F858[*(int *)a1] <= byte_452F858[*(int *)a2]) - 1;
  }
  return result;
}
