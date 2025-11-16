// Function: sub_667AD0
// Address: 0x667ad0
//
void __fastcall sub_667AD0(__int64 a1)
{
  __int64 v1; // r12
  char v2; // al
  __int64 v3; // rsi
  __int64 v4; // rdi

  v1 = *(_QWORD *)(a1 + 304);
  if ( v1 )
  {
    while ( 1 )
    {
      v2 = *(_BYTE *)(v1 + 140);
      if ( v2 != 12 )
        break;
      v1 = *(_QWORD *)(v1 + 160);
    }
    if ( v2 == 21 )
    {
      v3 = a1 + 104;
      if ( *(_QWORD *)a1 && *(_BYTE *)(*(_QWORD *)a1 + 80LL) == 7 )
      {
        sub_6851C0(2718, v3);
        goto LABEL_13;
      }
      v4 = 2714;
    }
    else
    {
      if ( *(char *)(a1 + 121) >= 0 )
        return;
      v3 = a1 + 104;
      v4 = 2715;
      if ( !v2 )
        return;
    }
    sub_6851C0(v4, v3);
LABEL_13:
    sub_725570(v1, 0);
  }
}
