// Function: sub_1444480
// Address: 0x1444480
//
__int64 __fastcall sub_1444480(__int64 a1)
{
  __int64 v1; // rax
  __int64 result; // rax
  int v3; // r12d

  if ( (unsigned int)sub_16AF4C0(&dword_4F9A048, 1, 0) )
  {
    do
    {
      v3 = dword_4F9A048;
      result = sub_16AF4B0();
    }
    while ( v3 != 2 );
  }
  else
  {
    sub_15CD350(a1);
    sub_1440EE0(a1);
    sub_13BF890(a1);
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 39;
      *(_QWORD *)v1 = "Detect single entry single exit regions";
      *(_QWORD *)(v1 + 16) = "regions";
      *(_QWORD *)(v1 + 24) = 7;
      *(_QWORD *)(v1 + 32) = &unk_4F9A04C;
      *(_WORD *)(v1 + 40) = 257;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_1444620;
    }
    sub_163A800(a1, v1, 1);
    result = sub_16AF4B0();
    dword_4F9A048 = 2;
  }
  return result;
}
