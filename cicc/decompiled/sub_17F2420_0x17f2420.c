// Function: sub_17F2420
// Address: 0x17f2420
//
void __fastcall sub_17F2420(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  int v5; // ebx
  int v6; // ebx

  if ( (unsigned int)sub_16AF4C0(&dword_4FA5A58, 1, 0) )
  {
    do
    {
      v5 = dword_4FA5A58;
      sub_16AF4B0();
      if ( v5 == 2 )
        break;
      v6 = dword_4FA5A58;
      sub_16AF4B0();
    }
    while ( v6 != 2 );
  }
  else
  {
    sub_1368E50(a1);
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 54;
      *(_QWORD *)v1 = "Optimize memory intrinsic using its size value profile";
      *(_QWORD *)(v1 + 16) = "pgo-memop-opt";
      *(_QWORD *)(v1 + 32) = &unk_4FA5A5C;
      *(_WORD *)(v1 + 40) = 0;
      *(_QWORD *)(v1 + 24) = 13;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_17F2510;
    }
    sub_163A800(a1, (_QWORD *)v1, 1, v2, v3, v4);
    sub_16AF4B0();
    dword_4FA5A58 = 2;
  }
}
