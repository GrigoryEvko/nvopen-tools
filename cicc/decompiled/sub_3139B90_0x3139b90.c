// Function: sub_3139B90
// Address: 0x3139b90
//
__int64 *__fastcall sub_3139B90(__int64 *a1, __int64 *a2)
{
  __int64 *result; // rax
  bool v3; // zf
  __int64 v4; // rdx

  result = a1;
  v3 = (a2[3] & 1) == 0;
  *((_BYTE *)a2 + 24) &= ~2u;
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
