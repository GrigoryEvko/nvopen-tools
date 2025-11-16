// Function: sub_149CB50
// Address: 0x149cb50
//
char __fastcall sub_149CB50(__int64 a1, __int64 a2, unsigned int *a3)
{
  __int64 v4; // r12
  _BYTE *v5; // rax
  __int64 v6; // rdx

  v4 = *(_QWORD *)(a2 + 40);
  if ( v4 )
    v4 = sub_1632FA0(*(_QWORD *)(a2 + 40));
  v5 = (_BYTE *)sub_1649960(a2);
  if ( (unsigned __int8)sub_149B630(a1, v5, v6, a3) )
    return sub_149B780(a1, *(__int64 **)(a2 + 24), *a3, v4);
  else
    return 0;
}
