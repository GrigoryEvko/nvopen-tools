// Function: sub_1F2F8A0
// Address: 0x1f2f8a0
//
void __fastcall sub_1F2F8A0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  int v5; // r12d

  if ( (unsigned int)sub_16AF4C0(&dword_4FCAC88, 1, 0) )
  {
    do
    {
      v5 = dword_4FCAC88;
      sub_16AF4B0();
    }
    while ( v5 != 2 );
  }
  else
  {
    sub_1F10320(a1);
    sub_1DC9950(a1);
    sub_1E29510(a1);
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 19;
      *(_QWORD *)v1 = "Stack Slot Coloring";
      *(_QWORD *)(v1 + 16) = "stack-slot-coloring";
      *(_QWORD *)(v1 + 32) = &unk_4FCAC8C;
      *(_WORD *)(v1 + 40) = 0;
      *(_QWORD *)(v1 + 24) = 19;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_1F2F990;
    }
    sub_163A800(a1, (_QWORD *)v1, 1, v2, v3, v4);
    sub_16AF4B0();
    dword_4FCAC88 = 2;
  }
}
