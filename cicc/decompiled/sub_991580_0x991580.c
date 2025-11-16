// Function: sub_991580
// Address: 0x991580
//
__int64 __fastcall sub_991580(__int64 a1, __int64 a2)
{
  _BYTE *v3; // rax

  if ( *(_BYTE *)a2 == 17 )
  {
    **(_QWORD **)a1 = a2 + 24;
    return 1;
  }
  else if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a2 + 8) + 8LL) - 17 <= 1
         && *(_BYTE *)a2 <= 0x15u
         && (v3 = (_BYTE *)sub_AD7630(a2, *(unsigned __int8 *)(a1 + 8))) != 0
         && *v3 == 17 )
  {
    **(_QWORD **)a1 = v3 + 24;
    return 1;
  }
  else
  {
    return 0;
  }
}
