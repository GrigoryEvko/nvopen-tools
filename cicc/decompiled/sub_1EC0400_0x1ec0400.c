// Function: sub_1EC0400
// Address: 0x1ec0400
//
void __fastcall sub_1EC0400(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  int v5; // r12d

  if ( (unsigned int)sub_16AF4C0(&dword_4FC91D4, 1, 0) )
  {
    do
    {
      v5 = dword_4FC91D4;
      sub_16AF4B0();
    }
    while ( v5 != 2 );
  }
  else
  {
    sub_1DA9400(a1);
    sub_1F10320(a1);
    sub_1DB9DF0(a1);
    sub_1EDA9C0(a1);
    sub_1E6EDE0(a1);
    sub_1DC9950(a1);
    sub_1E055C0(a1);
    sub_1E29510(a1);
    sub_1F5BA80(a1);
    sub_2102C30(a1);
    sub_20E9430(a1);
    sub_1F12110(a1);
    sub_1E36E00(a1);
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 25;
      *(_QWORD *)v1 = "Greedy Register Allocator";
      *(_QWORD *)(v1 + 16) = "greedy";
      *(_QWORD *)(v1 + 32) = &unk_4FC91D8;
      *(_WORD *)(v1 + 40) = 0;
      *(_QWORD *)(v1 + 24) = 6;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_1EBDCA0;
    }
    sub_163A800(a1, (_QWORD *)v1, 1, v2, v3, v4);
    sub_16AF4B0();
    dword_4FC91D4 = 2;
  }
}
