// Function: sub_5F04A0
// Address: 0x5f04a0
//
__int64 __fastcall sub_5F04A0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 i; // rdi

  for ( i = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  return sub_72F5E0(i, *(_QWORD *)(a1 + 64), a2, a3, a4, a5);
}
