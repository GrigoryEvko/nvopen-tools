// Function: sub_1A91D30
// Address: 0x1a91d30
//
void __fastcall sub_1A91D30(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  int v5; // r12d

  if ( (unsigned int)sub_16AF4C0(&dword_4FB57EC, 1, 0) )
  {
    do
    {
      v5 = dword_4FB57EC;
      sub_16AF4B0();
    }
    while ( v5 != 2 );
  }
  else
  {
    sub_1458320(a1);
    sub_15CD350(a1);
    sub_13FBE20(a1);
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 25;
      *(_QWORD *)v1 = "Place Backedge Safepoints";
      *(_QWORD *)(v1 + 16) = "place-backedge-safepoints-impl";
      *(_QWORD *)(v1 + 32) = &unk_4FB57F1;
      *(_WORD *)(v1 + 40) = 0;
      *(_QWORD *)(v1 + 24) = 30;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_1A91E20;
    }
    sub_163A800(a1, (_QWORD *)v1, 1, v2, v3, v4);
    sub_16AF4B0();
    dword_4FB57EC = 2;
  }
}
