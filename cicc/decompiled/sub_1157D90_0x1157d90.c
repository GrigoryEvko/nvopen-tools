// Function: sub_1157D90
// Address: 0x1157d90
//
__int64 __fastcall sub_1157D90(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE *v4; // rax

  if ( *(_BYTE *)a2 == 17 )
  {
    **(_QWORD **)a1 = a2 + 24;
    return 1;
  }
  else if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a2 + 8) + 8LL) - 17 <= 1
         && (v4 = sub_AD7630(a2, *(unsigned __int8 *)(a1 + 8), a3)) != 0
         && *v4 == 17 )
  {
    **(_QWORD **)a1 = v4 + 24;
    return 1;
  }
  else
  {
    return 0;
  }
}
