// Function: sub_1C73B00
// Address: 0x1c73b00
//
__int64 __fastcall sub_1C73B00(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rax
  unsigned int v5; // r8d
  __int64 v6; // rdx

  v4 = sub_157EBA0(a1);
  v5 = 0;
  if ( *(_BYTE *)(v4 + 16) == 26 && (*(_DWORD *)(v4 + 20) & 0xFFFFFFF) == 3 )
  {
    v6 = *(_QWORD *)(v4 - 24);
    if ( a2 == v6 && a3 == *(_QWORD *)(v4 - 48) )
    {
      return 1;
    }
    else
    {
      v5 = 0;
      if ( a3 == v6 )
        LOBYTE(v5) = *(_QWORD *)(v4 - 48) == a2;
    }
  }
  return v5;
}
