// Function: sub_2AAA3A0
// Address: 0x2aaa3a0
//
char __fastcall sub_2AAA3A0(_QWORD *a1, __int64 a2)
{
  char result; // al
  unsigned int v3; // edx

  result = *(_BYTE *)(a2 + 4);
  v3 = *(_DWORD *)a2;
  if ( result )
  {
    if ( v3 )
      return (unsigned int)sub_2AAA2B0(*(_QWORD *)(a1[1] + 48LL), *(_QWORD *)(*a1 + 48LL), v3, result) == 3;
    return 0;
  }
  else if ( v3 > 1 )
  {
    return (unsigned int)sub_2AAA2B0(*(_QWORD *)(a1[1] + 48LL), *(_QWORD *)(*a1 + 48LL), v3, result) == 3;
  }
  return result;
}
