// Function: sub_13B74A0
// Address: 0x13b74a0
//
__int64 __fastcall sub_13B74A0(__int64 a1)
{
  __int64 v1; // rax
  __int64 result; // rax
  int v3; // ebx
  int v4; // ebx

  if ( (unsigned int)sub_16AF4C0(&dword_4F98E18, 1, 0) )
  {
    do
    {
      v3 = dword_4F98E18;
      result = sub_16AF4B0();
      if ( v3 == 2 )
        break;
      v4 = dword_4F98E18;
      result = sub_16AF4B0();
    }
    while ( v4 != 2 );
  }
  else
  {
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 50;
      *(_QWORD *)v1 = "Print postdominance tree of function to 'dot' file";
      *(_QWORD *)(v1 + 16) = "dot-postdom";
      *(_QWORD *)(v1 + 32) = &unk_4F98E1C;
      *(_WORD *)(v1 + 40) = 0;
      *(_QWORD *)(v1 + 24) = 11;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_13B7580;
    }
    sub_163A800(a1, v1, 1);
    result = sub_16AF4B0();
    dword_4F98E18 = 2;
  }
  return result;
}
