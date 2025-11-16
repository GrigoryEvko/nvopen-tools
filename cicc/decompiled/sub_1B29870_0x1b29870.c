// Function: sub_1B29870
// Address: 0x1b29870
//
char __fastcall sub_1B29870(__int64 a1, __int64 a2, __int64 a3)
{
  if ( *(_QWORD *)(a2 + 40) == *(_QWORD *)(a3 + 40) )
    return sub_1B29560(a1, a2, a3);
  else
    return sub_15CC8F0(*(_QWORD *)(a1 + 32), *(_QWORD *)(a2 + 40), *(_QWORD *)(a3 + 40));
}
