// Function: sub_1117F70
// Address: 0x1117f70
//
void __fastcall sub_1117F70(__int64 a1, __int64 a2, char a3)
{
  char v3; // al
  __int64 v4; // r13
  __int16 v5; // dx
  __int64 v6; // r8
  char v7; // al
  char v8; // dl
  __int16 v9; // cx

  v3 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 <= 0x1Cu )
  {
    if ( v3 != 22 )
      return;
    v4 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 80LL);
    if ( v4 )
      v4 -= 24;
    goto LABEL_5;
  }
  if ( v3 == 84 )
  {
    v4 = *(_QWORD *)(a2 + 40);
LABEL_5:
    v6 = sub_AA5190(v4);
    if ( v6 )
    {
      v7 = v5;
      v8 = HIBYTE(v5);
    }
    else
    {
      v8 = 0;
      v7 = 0;
    }
    LOBYTE(v9) = v7;
    HIBYTE(v9) = v8;
    sub_A88F30(a1, v4, v6, v9);
    return;
  }
  if ( !a3 )
  {
    a2 = *(_QWORD *)(a2 + 32);
    if ( a2 )
      a2 -= 24;
  }
  sub_D5F1F0(a1, a2);
}
