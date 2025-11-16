// Function: sub_1304790
// Address: 0x1304790
//
__int64 __fastcall sub_1304790(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 144);
  if ( result )
  {
    result = sub_1317800(qword_50579C0[*(unsigned int *)(result + 78928)], 0);
    *(_QWORD *)(a1 + 144) = 0;
  }
  return result;
}
