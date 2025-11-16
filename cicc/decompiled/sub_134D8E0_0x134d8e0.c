// Function: sub_134D8E0
// Address: 0x134d8e0
//
__int64 __fastcall sub_134D8E0(__int64 a1)
{
  __int64 v1; // rax
  __int64 result; // rax
  int v3; // r12d

  if ( (unsigned int)sub_16AF4C0(&dword_4F96DB0, 1, 0) )
  {
    do
    {
      v3 = dword_4F96DB0;
      result = sub_16AF4B0();
    }
    while ( v3 != 2 );
  }
  else
  {
    sub_1361770(a1);
    sub_13838A0(a1);
    sub_138F5C0(a1);
    sub_134D700(a1);
    sub_13C2E30(a1);
    sub_1438670(a1);
    sub_1498BD0(a1);
    sub_149A4A0(a1);
    sub_14A7370(a1);
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 31;
      *(_QWORD *)v1 = "Function Alias Analysis Results";
      *(_QWORD *)(v1 + 16) = "aa";
      *(_QWORD *)(v1 + 24) = 2;
      *(_QWORD *)(v1 + 32) = &unk_4F96DB4;
      *(_WORD *)(v1 + 40) = 256;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_134DAC0;
    }
    sub_163A800(a1, v1, 1);
    result = sub_16AF4B0();
    dword_4F96DB0 = 2;
  }
  return result;
}
