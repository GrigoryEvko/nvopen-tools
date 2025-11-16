// Function: sub_1DE8060
// Address: 0x1de8060
//
void __fastcall sub_1DE8060(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  int v5; // r12d

  if ( (unsigned int)sub_16AF4C0(&dword_4FC4B08, 1, 0) )
  {
    do
    {
      v5 = dword_4FC4B08;
      sub_16AF4B0();
    }
    while ( v5 != 2 );
  }
  else
  {
    sub_1DF1590(a1);
    sub_1DDC080(a1);
    sub_1E5EC20(a1);
    sub_1E29510(a1);
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 40;
      *(_QWORD *)v1 = "Branch Probability Basic Block Placement";
      *(_QWORD *)(v1 + 16) = "block-placement";
      *(_QWORD *)(v1 + 32) = &unk_4FC4B0C;
      *(_WORD *)(v1 + 40) = 0;
      *(_QWORD *)(v1 + 24) = 15;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_1DE8160;
    }
    sub_163A800(a1, (_QWORD *)v1, 1, v2, v3, v4);
    sub_16AF4B0();
    dword_4FC4B08 = 2;
  }
}
