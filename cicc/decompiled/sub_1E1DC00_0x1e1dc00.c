// Function: sub_1E1DC00
// Address: 0x1e1dc00
//
void __fastcall sub_1E1DC00(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  int v5; // r12d

  if ( (unsigned int)sub_16AF4C0(&dword_4FC64C0, 1, 0) )
  {
    do
    {
      v5 = dword_4FC64C0;
      sub_16AF4B0();
    }
    while ( v5 != 2 );
  }
  else
  {
    sub_1E29510(a1);
    sub_1E055C0(a1);
    sub_134D8E0(a1);
    sub_21EAA00(a1);
    sub_1BFB430(a1);
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 40;
      *(_QWORD *)v1 = "Early Machine Loop Invariant Code Motion";
      *(_QWORD *)(v1 + 16) = "early-machinelicm";
      *(_QWORD *)(v1 + 32) = &unk_4FC64C8;
      *(_WORD *)(v1 + 40) = 0;
      *(_QWORD *)(v1 + 24) = 17;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_1E1DD00;
    }
    sub_163A800(a1, (_QWORD *)v1, 1, v2, v3, v4);
    sub_16AF4B0();
    dword_4FC64C0 = 2;
  }
}
