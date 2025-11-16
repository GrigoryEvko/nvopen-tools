// Function: sub_2AB31C0
// Address: 0x2ab31c0
//
__int64 __fastcall sub_2AB31C0(__int64 a1, char a2)
{
  __int64 result; // rax
  __int64 v3; // r13

  if ( *(_DWORD *)(a1 + 96) )
    return 0;
  v3 = sub_D46F00(*(_QWORD *)(a1 + 416));
  if ( v3 == sub_D47930(*(_QWORD *)(a1 + 416)) )
    goto LABEL_6;
  result = (unsigned __int8)byte_500CDA8;
  if ( !byte_500CDA8 )
    return 1;
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 440) + 664LL) )
  {
LABEL_6:
    result = 0;
    if ( a2 )
      return *(unsigned __int8 *)(*(_QWORD *)(a1 + 504) + 40LL);
  }
  return result;
}
