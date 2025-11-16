// Function: sub_1C76080
// Address: 0x1c76080
//
void __fastcall sub_1C76080(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  int v5; // r12d

  if ( (unsigned int)sub_16AF4C0(&dword_4FBD4A8, 1, 0) )
  {
    do
    {
      v5 = dword_4FBD4A8;
      sub_16AF4B0();
    }
    while ( v5 != 2 );
  }
  else
  {
    sub_15CD350(a1);
    sub_13FBE20(a1);
    sub_1AE1AE0(a1);
    sub_1AF93A0(a1);
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 17;
      *(_QWORD *)v1 = "Index Split Loops";
      *(_QWORD *)(v1 + 16) = "loop-index-split";
      *(_QWORD *)(v1 + 32) = &unk_4FBD4AC;
      *(_WORD *)(v1 + 40) = 0;
      *(_QWORD *)(v1 + 24) = 16;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_1C76180;
    }
    sub_163A800(a1, (_QWORD *)v1, 1, v2, v3, v4);
    sub_16AF4B0();
    dword_4FBD4A8 = 2;
  }
}
