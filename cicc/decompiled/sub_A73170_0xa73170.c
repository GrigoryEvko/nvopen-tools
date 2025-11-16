// Function: sub_A73170
// Address: 0xa73170
//
__int64 __fastcall sub_A73170(_QWORD *a1, int a2)
{
  __int64 result; // rax

  result = 0;
  if ( *a1 )
    return ((int)*(unsigned __int8 *)(*a1 + a2 / 8 + 12LL) >> (a2 % 8)) & 1;
  return result;
}
