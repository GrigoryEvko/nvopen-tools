// Function: sub_21F8260
// Address: 0x21f8260
//
__int64 __fastcall sub_21F8260(__int64 a1, __int64 a2)
{
  if ( sub_2162BB0(*(_QWORD *)(a1 + 496), a2) || (unsigned __int8)sub_2162BE0(*(_QWORD *)(a1 + 496), a2) )
    return 0;
  else
    return (*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 496) + 200LL))(*(_QWORD *)(a1 + 496), a2);
}
