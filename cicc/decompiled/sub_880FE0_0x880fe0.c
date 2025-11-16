// Function: sub_880FE0
// Address: 0x880fe0
//
__int64 __fastcall sub_880FE0(__int64 a1)
{
  __int64 i; // rax

  for ( i = *(_QWORD *)(a1 + 88); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  return sub_892920(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)i + 96LL) + 72LL));
}
