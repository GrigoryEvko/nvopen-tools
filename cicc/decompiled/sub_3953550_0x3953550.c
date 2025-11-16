// Function: sub_3953550
// Address: 0x3953550
//
void __fastcall sub_3953550(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  int v5; // ebx
  int v6; // ebx

  if ( (unsigned int)sub_16AF4C0(&dword_5054410, 1, 0) )
  {
    do
    {
      v5 = dword_5054410;
      sub_16AF4B0();
      if ( v5 == 2 )
        break;
      v6 = dword_5054410;
      sub_16AF4B0();
    }
    while ( v6 != 2 );
  }
  else
  {
    sub_13FBE20(a1);
    sub_15CD350(a1);
    v1 = sub_22077B0(0x50u);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 26;
      *(_QWORD *)v1 = "Register pressure analysis";
      *(_QWORD *)(v1 + 16) = "rpa";
      *(_QWORD *)(v1 + 24) = 3;
      *(_QWORD *)(v1 + 32) = &unk_5054414;
      *(_WORD *)(v1 + 40) = 256;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_3953650;
    }
    sub_163A800(a1, (_QWORD *)v1, 1, v2, v3, v4);
    sub_16AF4B0();
    dword_5054410 = 2;
  }
}
