// Function: sub_13F4FE0
// Address: 0x13f4fe0
//
__int64 __fastcall sub_13F4FE0(__int64 a1)
{
  __int64 v1; // rax
  __int64 result; // rax
  int v3; // r12d

  if ( (unsigned int)sub_16AF4C0(&dword_4F99134, 1, 0) )
  {
    do
    {
      v3 = dword_4F99134;
      result = sub_16AF4B0();
    }
    while ( v3 != 2 );
  }
  else
  {
    sub_14CAFD0(a1);
    sub_149CBF0(a1);
    sub_15CD350(a1);
    sub_134D8E0(a1);
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 30;
      *(_QWORD *)v1 = "Statically lint-checks LLVM IR";
      *(_QWORD *)(v1 + 16) = "lint";
      *(_QWORD *)(v1 + 32) = &unk_4F99138;
      *(_WORD *)(v1 + 40) = 256;
      *(_QWORD *)(v1 + 24) = 4;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_13F50E0;
    }
    sub_163A800(a1, v1, 1);
    result = sub_16AF4B0();
    dword_4F99134 = 2;
  }
  return result;
}
