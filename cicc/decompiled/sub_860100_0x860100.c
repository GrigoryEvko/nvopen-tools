// Function: sub_860100
// Address: 0x860100
//
__int64 __fastcall sub_860100(unsigned int a1)
{
  __int64 v1; // r12
  __int64 result; // rax
  __int64 i; // rbx

  if ( a1 )
    a1 = 8;
  sub_85C120(0, unk_4F066A8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, a1);
  v1 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  result = *(_QWORD *)(qword_4D03FF0 + 8);
  for ( i = *(_QWORD *)(result + 184); i; i = *(_QWORD *)i )
    result = sub_85EE10(i, v1, *(_DWORD *)(i + 56));
  return result;
}
