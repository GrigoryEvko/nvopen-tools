// Function: sub_3074700
// Address: 0x3074700
//
__int64 __fastcall sub_3074700(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rdi
  __int64 (*v4)(); // rdx
  __int64 result; // rax

  if ( *(_BYTE *)a2 != 85 )
    goto LABEL_5;
  if ( sub_CEA640(a2) )
    return 0;
  v2 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v2 != 25 )
  {
    if ( *(_BYTE *)v2
      || *(_QWORD *)(v2 + 24) != *(_QWORD *)(a2 + 80)
      || (sub_B2DCC0(*(_QWORD *)(a2 - 32)) || (unsigned __int8)sub_B2DCE0(v2)) && !sub_CEA1A0(*(_DWORD *)(v2 + 36)) )
    {
      goto LABEL_5;
    }
    return 0;
  }
  if ( *(_BYTE *)(v2 + 96) )
    return 0;
LABEL_5:
  v3 = *(_QWORD *)(a1 + 24);
  v4 = *(__int64 (**)())(*(_QWORD *)v3 + 1416LL);
  result = 1;
  if ( v4 != sub_2FE3490 )
    return ((__int64 (__fastcall *)(__int64, __int64))v4)(v3, a2);
  return result;
}
