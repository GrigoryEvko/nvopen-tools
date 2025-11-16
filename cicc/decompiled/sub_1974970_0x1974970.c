// Function: sub_1974970
// Address: 0x1974970
//
void __fastcall sub_1974970(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  int v5; // r12d

  if ( (unsigned int)sub_16AF4C0(&dword_4FB0730, 1, 0) )
  {
    do
    {
      v5 = dword_4FB0730;
      sub_16AF4B0();
    }
    while ( v5 != 2 );
  }
  else
  {
    sub_134D8E0(a1);
    sub_13A5DC0(a1);
    sub_15CD350(a1);
    sub_1458320(a1);
    sub_1AF93A0(a1);
    sub_1AE1AE0(a1);
    sub_13FBE20(a1);
    sub_143AAC0(a1);
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 34;
      *(_QWORD *)v1 = "Interchanges loops for cache reuse";
      *(_QWORD *)(v1 + 16) = "loop-interchange";
      *(_QWORD *)(v1 + 32) = &unk_4FB0734;
      *(_WORD *)(v1 + 40) = 0;
      *(_QWORD *)(v1 + 24) = 16;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_1974A90;
    }
    sub_163A800(a1, (_QWORD *)v1, 1, v2, v3, v4);
    sub_16AF4B0();
    dword_4FB0730 = 2;
  }
}
