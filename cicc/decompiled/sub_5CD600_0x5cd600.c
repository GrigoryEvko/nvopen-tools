// Function: sub_5CD600
// Address: 0x5cd600
//
__int64 __fastcall sub_5CD600(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  char v4; // al
  __int64 v6; // rax

  if ( (*(_BYTE *)(a2 + 40) & 1) != 0 )
  {
    v3 = unk_4F04C68 + 776LL * unk_4F04C64;
    v4 = *(_BYTE *)(v3 + 4);
    if ( (unsigned __int8)(v4 - 3) > 1u && v4 )
    {
      sub_6851C0(1300, a1 + 56);
      *(_BYTE *)(a1 + 8) = 0;
      return a2;
    }
    else
    {
      v6 = *(_QWORD *)(a2 + 24);
      *(_BYTE *)(a2 + 40) |= 0x60u;
      *(_BYTE *)(v6 + 124) |= 8u;
      sub_650490(v3, a2);
      return a2;
    }
  }
  else
  {
    sub_5CCAE0(8u, a1);
    return a2;
  }
}
