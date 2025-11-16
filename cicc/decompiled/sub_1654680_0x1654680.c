// Function: sub_1654680
// Address: 0x1654680
//
__int64 __fastcall sub_1654680(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 result; // rax
  int v6; // ebx
  int v7; // ebx

  if ( (unsigned int)sub_16AF4C0(&dword_4F9F048, 1, 0) )
  {
    do
    {
      v6 = dword_4F9F048;
      result = sub_16AF4B0();
      if ( v6 == 2 )
        break;
      v7 = dword_4F9F048;
      result = sub_16AF4B0();
    }
    while ( v7 != 2 );
  }
  else
  {
    v1 = sub_22077B0(80);
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 15;
      *(_QWORD *)v1 = "Module Verifier";
      *(_QWORD *)(v1 + 16) = "verify";
      *(_QWORD *)(v1 + 32) = &unk_4F9F04C;
      *(_WORD *)(v1 + 40) = 0;
      *(_QWORD *)(v1 + 24) = 6;
      *(_BYTE *)(v1 + 42) = 0;
      *(_QWORD *)(v1 + 48) = 0;
      *(_QWORD *)(v1 + 56) = 0;
      *(_QWORD *)(v1 + 64) = 0;
      *(_QWORD *)(v1 + 72) = sub_1654760;
    }
    sub_163A800(a1, (_QWORD *)v1, 1, v2, v3, v4);
    result = sub_16AF4B0();
    dword_4F9F048 = 2;
  }
  return result;
}
