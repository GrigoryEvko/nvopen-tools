// Function: sub_18F18F0
// Address: 0x18f18f0
//
void __fastcall sub_18F18F0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  int v5; // ebx
  int v6; // ebx

  if ( (unsigned int)sub_16AF4C0(&dword_4FAE350, 1, 0) )
  {
    do
    {
      v5 = dword_4FAE350;
      sub_16AF4B0();
      if ( v5 == 2 )
        break;
      v6 = dword_4FAE350;
      sub_16AF4B0();
    }
    while ( v6 != 2 );
  }
  else
  {
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 28;
      *(_QWORD *)v1 = "Dead Instruction Elimination";
      *(_QWORD *)(v1 + 16) = "die";
      *(_QWORD *)(v1 + 32) = &unk_4FAE354;
      *(_WORD *)(v1 + 40) = 0;
      *(_QWORD *)(v1 + 24) = 3;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_18F19D0;
    }
    sub_163A800(a1, (_QWORD *)v1, 1, v2, v3, v4);
    sub_16AF4B0();
    dword_4FAE350 = 2;
  }
}
