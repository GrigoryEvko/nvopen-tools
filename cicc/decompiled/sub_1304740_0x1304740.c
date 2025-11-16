// Function: sub_1304740
// Address: 0x1304740
//
__int64 __fastcall sub_1304740(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 136);
  if ( result )
  {
    result = sub_1317800(qword_50579C0[*(unsigned int *)(result + 78928)], 1);
    *(_QWORD *)(a1 + 136) = 0;
  }
  return result;
}
