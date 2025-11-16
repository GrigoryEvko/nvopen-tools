// Function: sub_1E127E0
// Address: 0x1e127e0
//
void __fastcall sub_1E127E0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  int v5; // ebx
  int v6; // ebx

  if ( (unsigned int)sub_16AF4C0(&dword_4FC64A8, 1, 0) )
  {
    do
    {
      v5 = dword_4FC64A8;
      sub_16AF4B0();
      if ( v5 == 2 )
        break;
      v6 = dword_4FC64A8;
      sub_16AF4B0();
    }
    while ( v6 != 2 );
  }
  else
  {
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 24;
      *(_QWORD *)v1 = "Machine Function Printer";
      *(_QWORD *)(v1 + 16) = "machineinstr-printer";
      *(_QWORD *)(v1 + 32) = &unk_4FC64AC;
      *(_WORD *)(v1 + 40) = 0;
      *(_QWORD *)(v1 + 24) = 20;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_1E12580;
    }
    sub_163A800(a1, (_QWORD *)v1, 1, v2, v3, v4);
    sub_16AF4B0();
    dword_4FC64A8 = 2;
  }
}
