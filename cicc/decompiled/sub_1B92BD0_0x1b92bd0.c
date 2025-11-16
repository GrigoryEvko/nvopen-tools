// Function: sub_1B92BD0
// Address: 0x1b92bd0
//
void __fastcall sub_1B92BD0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  int v5; // r12d

  if ( (unsigned int)sub_16AF4C0(&dword_4FB7EA8, 1, 0) )
  {
    do
    {
      v5 = dword_4FB7EA8;
      sub_16AF4B0();
    }
    while ( v5 != 2 );
  }
  else
  {
    sub_14A3D70(a1);
    sub_1361770(a1);
    sub_134D8E0(a1);
    sub_13C2E30(a1);
    sub_14CAFD0(a1);
    sub_1368E50(a1);
    sub_15CD350(a1);
    sub_1458320(a1);
    sub_13FBE20(a1);
    sub_3862A40(a1);
    sub_139F770(a1);
    sub_143AAC0(a1);
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 18;
      *(_QWORD *)v1 = "Loop Vectorization";
      *(_QWORD *)(v1 + 16) = "loop-vectorize";
      *(_QWORD *)(v1 + 32) = &unk_4FB7EAC;
      *(_WORD *)(v1 + 40) = 0;
      *(_QWORD *)(v1 + 24) = 14;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_1B92D10;
    }
    sub_163A800(a1, (_QWORD *)v1, 1, v2, v3, v4);
    sub_16AF4B0();
    dword_4FB7EA8 = 2;
  }
}
