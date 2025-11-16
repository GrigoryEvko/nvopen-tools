// Function: sub_217DBF0
// Address: 0x217dbf0
//
void __fastcall sub_217DBF0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  int v5; // ebx
  int v6; // ebx

  if ( (unsigned int)sub_16AF4C0(&dword_4FD2B28, 1, 0) )
  {
    do
    {
      v5 = dword_4FD2B28;
      sub_16AF4B0();
      if ( v5 == 2 )
        break;
      v6 = dword_4FD2B28;
      sub_16AF4B0();
    }
    while ( v6 != 2 );
  }
  else
  {
    sub_1E29510(a1);
    sub_1BFB430(a1);
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 26;
      *(_QWORD *)v1 = "NVPTX Specific Block Remat";
      *(_QWORD *)(v1 + 16) = "nvptx-remat-block";
      *(_QWORD *)(v1 + 32) = &unk_4CD4B10;
      *(_WORD *)(v1 + 40) = 0;
      *(_QWORD *)(v1 + 24) = 17;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_217DCE0;
    }
    sub_163A800(a1, (_QWORD *)v1, 1, v2, v3, v4);
    sub_16AF4B0();
    dword_4FD2B28 = 2;
  }
}
