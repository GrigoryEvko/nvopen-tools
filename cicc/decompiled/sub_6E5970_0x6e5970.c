// Function: sub_6E5970
// Address: 0x6e5970
//
void __fastcall sub_6E5970(__int64 a1)
{
  char v2; // al
  __int64 v3; // rax

LABEL_1:
  if ( a1 )
  {
    do
    {
      v2 = *(_BYTE *)(a1 + 8);
      if ( v2 )
      {
        if ( v2 == 1 )
          sub_6E5970(*(_QWORD *)(a1 + 24));
      }
      else
      {
        sub_6E5940(*(_QWORD *)(a1 + 24) + 8LL);
      }
      v3 = *(_QWORD *)a1;
      if ( !*(_QWORD *)a1 )
        break;
      if ( *(_BYTE *)(v3 + 8) == 3 )
      {
        a1 = sub_6BBB10((_QWORD *)a1);
        goto LABEL_1;
      }
      a1 = *(_QWORD *)a1;
    }
    while ( v3 );
  }
}
