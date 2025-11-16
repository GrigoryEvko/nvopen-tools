// Function: sub_1AAC350
// Address: 0x1aac350
//
void __fastcall sub_1AAC350(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  int v5; // ebx
  int v6; // ebx

  if ( (unsigned int)sub_16AF4C0(&dword_4FB6268, 1, 0) )
  {
    do
    {
      v5 = dword_4FB6268;
      sub_16AF4B0();
      if ( v5 == 2 )
        break;
      v6 = dword_4FB6268;
      sub_16AF4B0();
    }
    while ( v6 != 2 );
  }
  else
  {
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 27;
      *(_QWORD *)v1 = "Break critical edges in CFG";
      *(_QWORD *)(v1 + 16) = "break-crit-edges";
      *(_QWORD *)(v1 + 32) = &unk_4FB626C;
      *(_WORD *)(v1 + 40) = 0;
      *(_QWORD *)(v1 + 24) = 16;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_1AAC430;
    }
    sub_163A800(a1, (_QWORD *)v1, 1, v2, v3, v4);
    sub_16AF4B0();
    dword_4FB6268 = 2;
  }
}
