// Function: sub_13EAF90
// Address: 0x13eaf90
//
__int64 __fastcall sub_13EAF90(__int64 a1)
{
  __int64 v1; // rax
  __int64 result; // rax
  int v3; // ebx
  int v4; // ebx

  if ( (unsigned int)sub_16AF4C0(&dword_4F9912C, 1, 0) )
  {
    do
    {
      v3 = dword_4F9912C;
      result = sub_16AF4B0();
      if ( v3 == 2 )
        break;
      v4 = dword_4F9912C;
      result = sub_16AF4B0();
    }
    while ( v4 != 2 );
  }
  else
  {
    sub_14CAFD0(a1);
    sub_149CBF0(a1);
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 31;
      *(_QWORD *)v1 = "Lazy Value Information Analysis";
      *(_QWORD *)(v1 + 16) = "lazy-value-info";
      *(_QWORD *)(v1 + 24) = 15;
      *(_QWORD *)(v1 + 32) = &unk_4F99130;
      *(_WORD *)(v1 + 40) = 256;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_13EB090;
    }
    sub_163A800(a1, v1, 1);
    result = sub_16AF4B0();
    dword_4F9912C = 2;
  }
  return result;
}
