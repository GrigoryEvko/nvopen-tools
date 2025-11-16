// Function: sub_19C9140
// Address: 0x19c9140
//
void __fastcall sub_19C9140(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  int v5; // r12d

  if ( (unsigned int)sub_16AF4C0(&dword_4FB3748, 1, 0) )
  {
    do
    {
      v5 = dword_4FB3748;
      sub_16AF4B0();
    }
    while ( v5 != 2 );
  }
  else
  {
    sub_134D8E0(a1);
    sub_15CD350(a1);
    sub_13C2E30(a1);
    sub_1AE1AE0(a1);
    sub_3862A40(a1);
    sub_13FBE20(a1);
    sub_1AF93A0(a1);
    sub_1458320(a1);
    sub_143AAC0(a1);
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 24;
      *(_QWORD *)v1 = "Loop Versioning For LICM";
      *(_QWORD *)(v1 + 16) = "loop-versioning-licm";
      *(_QWORD *)(v1 + 32) = &unk_4FB374C;
      *(_WORD *)(v1 + 40) = 0;
      *(_QWORD *)(v1 + 24) = 20;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_19C9260;
    }
    sub_163A800(a1, (_QWORD *)v1, 1, v2, v3, v4);
    sub_16AF4B0();
    dword_4FB3748 = 2;
  }
}
