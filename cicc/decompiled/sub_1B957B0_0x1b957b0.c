// Function: sub_1B957B0
// Address: 0x1b957b0
//
_QWORD *__fastcall sub_1B957B0(_QWORD *a1, __int64 *a2)
{
  __int64 v2; // rdx
  _QWORD *result; // rax
  __int64 v4; // rcx
  __int64 v5; // rdx
  __int64 v6; // rdx

  v2 = a2[2];
  result = a1;
  if ( v2 == a2[1] )
    v4 = *((unsigned int *)a2 + 7);
  else
    v4 = *((unsigned int *)a2 + 6);
  v5 = v2 + 8 * v4;
  *a1 = v5;
  a1[1] = v5;
  v6 = *a2;
  a1[2] = a2;
  a1[3] = v6;
  return result;
}
