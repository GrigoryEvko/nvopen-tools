// Function: sub_72F530
// Address: 0x72f530
//
__int64 __fastcall sub_72F530(__int64 a1)
{
  __int64 i; // rax

  for ( i = *(_QWORD *)(a1 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  return sub_8D3110(*(_QWORD *)(**(_QWORD **)(i + 168) + 8LL));
}
