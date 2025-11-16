// Function: sub_2D611F0
// Address: 0x2d611f0
//
__int64 *__fastcall sub_2D611F0(__int64 *a1, __int64 a2)
{
  __int64 *result; // rax
  __int64 v3; // rcx
  __int64 v4; // rdx
  __int64 v5; // rdx

  result = a1;
  if ( *(_BYTE *)(a2 + 28) )
    v3 = *(unsigned int *)(a2 + 20);
  else
    v3 = *(unsigned int *)(a2 + 16);
  v4 = *(_QWORD *)(a2 + 8) + 8 * v3;
  *a1 = v4;
  a1[1] = v4;
  v5 = *(_QWORD *)a2;
  a1[2] = a2;
  a1[3] = v5;
  return result;
}
