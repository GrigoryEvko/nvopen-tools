// Function: sub_1361770
// Address: 0x1361770
//
__int64 __fastcall sub_1361770(__int64 a1)
{
  __int64 v1; // rax
  __int64 result; // rax
  int v3; // r12d

  if ( (unsigned int)sub_16AF4C0(&dword_4F97BA8, 1, 0) )
  {
    do
    {
      v3 = dword_4F97BA8;
      result = sub_16AF4B0();
    }
    while ( v3 != 2 );
  }
  else
  {
    sub_14CAFD0(a1);
    sub_15CD350(a1);
    sub_149CBF0(a1);
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 40;
      *(_QWORD *)v1 = "Basic Alias Analysis (stateless AA impl)";
      *(_QWORD *)(v1 + 16) = "basicaa";
      *(_QWORD *)(v1 + 24) = 7;
      *(_QWORD *)(v1 + 32) = &unk_4F97BAC;
      *(_WORD *)(v1 + 40) = 256;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_1361920;
    }
    sub_163A800(a1, v1, 1);
    result = sub_16AF4B0();
    dword_4F97BA8 = 2;
  }
  return result;
}
