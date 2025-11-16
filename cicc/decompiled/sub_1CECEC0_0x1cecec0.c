// Function: sub_1CECEC0
// Address: 0x1cecec0
//
void __fastcall sub_1CECEC0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  int v5; // ebx
  int v6; // ebx

  if ( (unsigned int)sub_16AF4C0(&dword_4FC0638, 1, 0) )
  {
    do
    {
      v5 = dword_4FC0638;
      sub_16AF4B0();
      if ( v5 == 2 )
        break;
      v6 = dword_4FC0638;
      sub_16AF4B0();
    }
    while ( v6 != 2 );
  }
  else
  {
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 41;
      *(_QWORD *)v1 = "Libdevice library specific checking phase";
      *(_QWORD *)(v1 + 16) = "nvvm-libdevice-check";
      *(_QWORD *)(v1 + 24) = 20;
      *(_QWORD *)(v1 + 32) = &unk_4CD49B8;
      *(_WORD *)(v1 + 40) = 0;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_1CECDB0;
    }
    sub_163A800(a1, (_QWORD *)v1, 1, v2, v3, v4);
    sub_16AF4B0();
    dword_4FC0638 = 2;
  }
}
