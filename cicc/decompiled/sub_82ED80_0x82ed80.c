// Function: sub_82ED80
// Address: 0x82ed80
//
__int64 __fastcall sub_82ED80(__int64 a1)
{
  __int64 v2; // rax
  char v3; // al
  __int64 v4; // rbx

  while ( a1 )
  {
    v3 = *(_BYTE *)(a1 + 8);
    if ( !v3 )
    {
      v4 = *(_QWORD *)(a1 + 24);
      if ( (unsigned int)sub_696840(v4 + 8) || *(_BYTE *)(v4 + 24) == 2 && *(_BYTE *)(v4 + 325) == 12 )
        return 1;
      goto LABEL_4;
    }
    if ( v3 != 1 )
    {
      if ( v3 != 2 )
        sub_721090();
LABEL_4:
      v2 = *(_QWORD *)a1;
      if ( !*(_QWORD *)a1 )
        return 0;
      goto LABEL_5;
    }
    if ( (unsigned int)sub_82ED80(*(_QWORD *)(a1 + 24)) )
      return 1;
    v2 = *(_QWORD *)a1;
    if ( !*(_QWORD *)a1 )
      return 0;
LABEL_5:
    if ( *(_BYTE *)(v2 + 8) == 3 )
      a1 = sub_6BBB10((_QWORD *)a1);
    else
      a1 = v2;
  }
  return 0;
}
