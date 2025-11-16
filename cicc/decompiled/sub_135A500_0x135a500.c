// Function: sub_135A500
// Address: 0x135a500
//
__int64 __fastcall sub_135A500(__int64 a1)
{
  __int64 v1; // rax
  __int64 result; // rax
  int v3; // ebx
  int v4; // ebx

  if ( (unsigned int)sub_16AF4C0(&dword_4F97AC8, 1, 0) )
  {
    do
    {
      v3 = dword_4F97AC8;
      result = sub_16AF4B0();
      if ( v3 == 2 )
        break;
      v4 = dword_4F97AC8;
      result = sub_16AF4B0();
    }
    while ( v4 != 2 );
  }
  else
  {
    sub_134D8E0(a1);
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 17;
      *(_QWORD *)v1 = "Alias Set Printer";
      *(_QWORD *)(v1 + 16) = "print-alias-sets";
      *(_QWORD *)(v1 + 32) = &unk_4F97ACC;
      *(_WORD *)(v1 + 40) = 256;
      *(_QWORD *)(v1 + 24) = 16;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_135A5F0;
    }
    sub_163A800(a1, v1, 1);
    result = sub_16AF4B0();
    dword_4F97AC8 = 2;
  }
  return result;
}
