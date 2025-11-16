// Function: sub_1649AF0
// Address: 0x1649af0
//
__int64 __fastcall sub_1649AF0(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  unsigned __int64 *v3; // rdx

  a1[1] = *(_QWORD *)(a2 + 8);
  result = (a2 + 8) | *a1 & 7;
  *a1 = result;
  *(_QWORD *)(a2 + 8) = a1;
  v3 = (unsigned __int64 *)a1[1];
  if ( v3 )
  {
    result = *v3 & 7;
    *v3 = result | (unsigned __int64)(a1 + 1);
  }
  return result;
}
