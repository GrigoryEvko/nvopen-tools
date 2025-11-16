// Function: sub_39EFF60
// Address: 0x39eff60
//
void __fastcall sub_39EFF60(__int64 a1, __int64 a2)
{
  unsigned int v3; // eax
  unsigned __int16 v4; // ax

  while ( 1 )
  {
    while ( 1 )
    {
      v3 = *(_DWORD *)a2;
      if ( *(_DWORD *)a2 != 3 )
        break;
      a2 = *(_QWORD *)(a2 + 24);
    }
    if ( v3 > 3 )
    {
      if ( v3 == 4 )
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)(a2 - 8) + 72LL))(a2 - 8, *(_QWORD *)(a1 + 264));
      return;
    }
    if ( v3 )
      break;
    sub_39EFF60(a1, *(_QWORD *)(a2 + 24));
    a2 = *(_QWORD *)(a2 + 32);
  }
  if ( v3 != 2 )
    return;
  v4 = *(_WORD *)(a2 + 16);
  if ( v4 <= 0x62u )
  {
    if ( v4 <= 0x3Cu )
    {
      if ( v4 <= 9u )
      {
        if ( v4 <= 5u )
          return;
      }
      else if ( (unsigned __int16)(v4 - 11) > 4u )
      {
        return;
      }
    }
    goto LABEL_13;
  }
  if ( (unsigned __int16)(v4 - 118) <= 1u )
  {
LABEL_13:
    sub_390D5F0(*(_QWORD *)(a1 + 264), *(_QWORD *)(a2 + 24), 0);
    sub_38E28A0(*(_QWORD *)(a2 + 24), 6);
  }
}
