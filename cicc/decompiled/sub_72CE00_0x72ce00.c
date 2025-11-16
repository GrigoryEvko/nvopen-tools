// Function: sub_72CE00
// Address: 0x72ce00
//
__int64 __fastcall sub_72CE00(__int64 a1)
{
  char v1; // al
  __int64 i; // rdx
  __int64 v3; // rcx
  char v5; // al
  const char *v6; // rsi

  v1 = *(_BYTE *)(a1 + 140);
  for ( i = a1; v1 == 12; v1 = *(_BYTE *)(i + 140) )
    i = *(_QWORD *)(i + 160);
  if ( qword_4F07B50 == i )
    return 1;
  if ( qword_4F07B50 )
  {
    sub_6851C0(0xD22u, dword_4F07508);
    return 0;
  }
  else
  {
    v3 = *(_QWORD *)(i + 40);
    if ( v3
      && *(_BYTE *)(v3 + 28) == 3
      && *(_QWORD *)(v3 + 32) == qword_4D049B8[11]
      && (unsigned __int8)(v1 - 9) <= 2u
      && (v5 = *(_BYTE *)(i + 89), (v5 & 0x40) == 0)
      && ((v5 & 8) == 0 ? (v6 = *(const char **)(i + 8)) : (v6 = *(const char **)(i + 24)),
          v6 && !strcmp(v6, "basic_string_view")) )
    {
      qword_4F07B50 = i;
      return 1;
    }
    else
    {
      sub_685360(0xD21u, dword_4F07508, i);
      return 0;
    }
  }
}
