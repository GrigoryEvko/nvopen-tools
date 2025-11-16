// Function: sub_6E9F70
// Address: 0x6e9f70
//
void __fastcall sub_6E9F70(__int64 a1)
{
  __int64 v2; // rax
  char v3; // al
  bool v4; // zf

  while ( a1 )
  {
    v3 = *(_BYTE *)(a1 + 9);
    if ( (v3 & 0x20) != 0 || (v4 = *(_BYTE *)(a1 + 8) == 1, *(_BYTE *)(a1 + 9) = v3 | 0x20, !v4) )
    {
      v2 = *(_QWORD *)a1;
      if ( !*(_QWORD *)a1 )
        return;
    }
    else
    {
      sub_6E9F70(*(_QWORD *)(a1 + 24));
      v2 = *(_QWORD *)a1;
      if ( !*(_QWORD *)a1 )
        return;
    }
    if ( *(_BYTE *)(v2 + 8) == 3 )
      a1 = sub_6BBB10((_QWORD *)a1);
    else
      a1 = v2;
  }
}
