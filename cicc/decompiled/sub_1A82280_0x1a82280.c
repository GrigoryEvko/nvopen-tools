// Function: sub_1A82280
// Address: 0x1a82280
//
void __fastcall sub_1A82280(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  int v5; // ebx
  int v6; // ebx

  if ( (unsigned int)sub_16AF4C0(&dword_4FB4F28, 1, 0) )
  {
    do
    {
      v5 = dword_4FB4F28;
      sub_16AF4B0();
      if ( v5 == 2 )
        break;
      v6 = dword_4FB4F28;
      sub_16AF4B0();
    }
    while ( v6 != 2 );
  }
  else
  {
    sub_134D8E0(a1);
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 15;
      *(_QWORD *)v1 = "Flatten the CFG";
      *(_QWORD *)(v1 + 16) = "flattencfg";
      *(_QWORD *)(v1 + 32) = &unk_4FB4F2C;
      *(_WORD *)(v1 + 40) = 0;
      *(_QWORD *)(v1 + 24) = 10;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_1A82370;
    }
    sub_163A800(a1, (_QWORD *)v1, 1, v2, v3, v4);
    sub_16AF4B0();
    dword_4FB4F28 = 2;
  }
}
