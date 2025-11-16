// Function: sub_39512D0
// Address: 0x39512d0
//
void __fastcall sub_39512D0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  int v5; // ebx
  int v6; // ebx

  if ( (unsigned int)sub_16AF4C0(&dword_5054408, 1, 0) )
  {
    do
    {
      v5 = dword_5054408;
      sub_16AF4B0();
      if ( v5 == 2 )
        break;
      v6 = dword_5054408;
      sub_16AF4B0();
    }
    while ( v6 != 2 );
  }
  else
  {
    sub_15CD350(a1);
    v1 = sub_22077B0(0x50u);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 10;
      *(_QWORD *)v1 = "Merge sets";
      *(_QWORD *)(v1 + 16) = "merge-sets";
      *(_QWORD *)(v1 + 24) = 10;
      *(_QWORD *)(v1 + 32) = &unk_505440C;
      *(_WORD *)(v1 + 40) = 257;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_3951480;
    }
    sub_163A800(a1, (_QWORD *)v1, 1, v2, v3, v4);
    sub_16AF4B0();
    dword_5054408 = 2;
  }
}
