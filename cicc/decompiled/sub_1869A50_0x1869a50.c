// Function: sub_1869A50
// Address: 0x1869a50
//
void __fastcall sub_1869A50(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  int v5; // r12d

  if ( (unsigned int)sub_16AF4C0(&dword_4FAB3F8, 1, 0) )
  {
    do
    {
      v5 = dword_4FAB3F8;
      sub_16AF4B0();
    }
    while ( v5 != 2 );
  }
  else
  {
    sub_14CAFD0(a1);
    sub_1398760(a1);
    sub_1442370(a1);
    sub_14A3D70(a1);
    sub_149CBF0(a1);
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 29;
      *(_QWORD *)v1 = "Function Integration/Inlining";
      *(_QWORD *)(v1 + 16) = "inline";
      *(_QWORD *)(v1 + 32) = &unk_4FAB3FC;
      *(_WORD *)(v1 + 40) = 0;
      *(_QWORD *)(v1 + 24) = 6;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_1869B50;
    }
    sub_163A800(a1, (_QWORD *)v1, 1, v2, v3, v4);
    sub_16AF4B0();
    dword_4FAB3F8 = 2;
  }
}
