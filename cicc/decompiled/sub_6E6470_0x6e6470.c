// Function: sub_6E6470
// Address: 0x6e6470
//
void __fastcall sub_6E6470(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rax
  char v3; // al

  v1 = a1;
  while ( v1 )
  {
    v3 = *(_BYTE *)(v1 + 8);
    if ( v3 )
    {
      if ( v3 == 1 )
      {
        a1 = *(_QWORD *)(v1 + 24);
        sub_6E6470(a1);
      }
      else if ( v3 != 2 )
      {
        sub_721090(a1);
      }
      v2 = *(_QWORD *)v1;
      if ( !*(_QWORD *)v1 )
        return;
    }
    else
    {
      a1 = *(_QWORD *)(v1 + 24) + 8LL;
      sub_6E6450(a1);
      v2 = *(_QWORD *)v1;
      if ( !*(_QWORD *)v1 )
        return;
    }
    if ( *(_BYTE *)(v2 + 8) == 3 )
    {
      a1 = v1;
      v1 = sub_6BBB10((_QWORD *)v1);
    }
    else
    {
      v1 = v2;
    }
  }
}
