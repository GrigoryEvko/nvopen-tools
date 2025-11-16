// Function: sub_9C94E0
// Address: 0x9c94e0
//
__int64 *__fastcall sub_9C94E0(__int64 *a1, __int64 *a2)
{
  __int64 *result; // rax
  bool v3; // zf
  __int64 v4; // rdx

  result = a1;
  v3 = (a2[4] & 1) == 0;
  *((_BYTE *)a2 + 32) &= ~2u;
  if ( v3 )
  {
    *a1 = 1;
  }
  else
  {
    v4 = *a2;
    *a2 = 0;
    *a1 = v4 | 1;
  }
  return result;
}
