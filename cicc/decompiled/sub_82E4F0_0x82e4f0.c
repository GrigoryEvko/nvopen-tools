// Function: sub_82E4F0
// Address: 0x82e4f0
//
void __fastcall sub_82E4F0(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v5; // r12
  __int64 v6; // rdi

  if ( a2 )
  {
    v5 = a2;
    sub_7461E0((__int64)&qword_4F5F780);
    v6 = a2;
    qword_4F5F780 = (__int64)sub_729610;
    byte_4F5F811 = dword_4F07460;
    qword_4F06C40 = 0;
    while ( v6 )
    {
      if ( *(_BYTE *)(v6 + 8) == 2 )
      {
        sub_7295A0("initializer list with designators");
        goto LABEL_16;
      }
      if ( !*(_QWORD *)v6 )
        break;
      if ( *(_BYTE *)(*(_QWORD *)v6 + 8LL) == 3 )
        v6 = sub_6BBB10((_QWORD *)v6);
      else
        v6 = *(_QWORD *)v6;
    }
    while ( v5 )
    {
      if ( *(_BYTE *)(v5 + 8) )
      {
        sub_7295A0("{...}");
        if ( !*(_QWORD *)v5 )
          break;
      }
      else
      {
        sub_74B930(*(_QWORD *)(*(_QWORD *)(v5 + 24) + 8LL), (__int64)&qword_4F5F780);
        if ( !*(_QWORD *)v5 )
          break;
      }
      sub_7295A0(", ");
      if ( !*(_QWORD *)v5 )
        break;
      if ( *(_BYTE *)(*(_QWORD *)v5 + 8LL) == 3 )
        v5 = sub_6BBB10((_QWORD *)v5);
      else
        v5 = *(_QWORD *)v5;
    }
LABEL_16:
    sub_729660(0);
    sub_67DCF0(a3, 739, (__int64)qword_4F06C50);
  }
  if ( a1 )
    sub_82E460(a1, a3);
}
