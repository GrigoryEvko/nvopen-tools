// Function: sub_1A6D600
// Address: 0x1a6d600
//
void __fastcall sub_1A6D600(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  int v5; // r12d

  if ( (unsigned int)sub_16AF4C0(&dword_4FB4D70, 1, 0) )
  {
    do
    {
      v5 = dword_4FB4D70;
      sub_16AF4B0();
    }
    while ( v5 != 2 );
  }
  else
  {
    sub_13B36A0(a1);
    sub_1B24840(a1);
    sub_15CD350(a1);
    sub_1444480(a1);
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 19;
      *(_QWORD *)v1 = "Structurize the CFG";
      *(_QWORD *)(v1 + 16) = "structurizecfg";
      *(_QWORD *)(v1 + 32) = &unk_4FB4D74;
      *(_WORD *)(v1 + 40) = 0;
      *(_QWORD *)(v1 + 24) = 14;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_1A6D9E0;
    }
    sub_163A800(a1, (_QWORD *)v1, 1, v2, v3, v4);
    sub_16AF4B0();
    dword_4FB4D70 = 2;
  }
}
