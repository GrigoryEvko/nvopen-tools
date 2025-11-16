// Function: sub_3089AC0
// Address: 0x3089ac0
//
__int64 __fastcall sub_3089AC0(__int64 a1, __int64 a2)
{
  if ( (unsigned __int8)sub_36D6600(*(_QWORD *)(a1 + 464)) || (unsigned __int8)sub_36D6610(*(_QWORD *)(a1 + 464), a2) )
    return 0;
  else
    return (*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 464) + 1600LL))(*(_QWORD *)(a1 + 464), a2);
}
