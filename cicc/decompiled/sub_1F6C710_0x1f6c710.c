// Function: sub_1F6C710
// Address: 0x1f6c710
//
__int64 __fastcall sub_1F6C710(unsigned int *a1, int a2)
{
  __int64 result; // rax
  __int64 v3; // r8
  unsigned int *v4; // rcx
  unsigned int *v5; // rcx
  __int64 v6; // rsi

  if ( !a2 )
    return 0;
  result = *(_QWORD *)a1;
  if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 40LL) + 16LL * a1[2]) == 1 )
    return result;
  v3 = (unsigned int)(a2 - 1);
  v4 = &a1[10 * v3];
  result = *(_QWORD *)v4;
  if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v4 + 40LL) + 16LL * v4[2]) == 1 )
    return result;
  if ( (unsigned int)v3 <= 1 )
    return 0;
  v5 = a1 + 10;
  v6 = (__int64)&a1[10 * (a2 - 3) + 20];
  while ( 1 )
  {
    result = *(_QWORD *)v5;
    if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v5 + 40LL) + 16LL * v5[2]) == 1 )
      break;
    v5 += 10;
    if ( (unsigned int *)v6 == v5 )
      return 0;
  }
  return result;
}
