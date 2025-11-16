// Function: sub_2EBE590
// Address: 0x2ebe590
//
__int64 __fastcall sub_2EBE590(__int64 a1, int a2, __int64 a3, unsigned int a4)
{
  if ( (unsigned int)(a2 - 1) <= 0x3FFFFFFE )
    return 0;
  else
    return sub_2EBE500(
             a1,
             a2,
             *(_QWORD *)(*(_QWORD *)(a1 + 56) + 16LL * (a2 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
             a3,
             a4);
}
