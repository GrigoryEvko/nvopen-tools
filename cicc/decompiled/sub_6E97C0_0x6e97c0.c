// Function: sub_6E97C0
// Address: 0x6e97c0
//
__int64 __fastcall sub_6E97C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v4; // al
  __int64 v6; // rax

  v4 = *(_BYTE *)(a1 + 16);
  if ( v4 == 1 )
  {
    v6 = *(_QWORD *)(a1 + 144);
    if ( *(_BYTE *)(v6 + 24) != 2 )
      return 0;
    return (unsigned int)sub_72A2A0(*(_QWORD *)(v6 + 56), a2, a3, a4, 0) != 0;
  }
  else
  {
    if ( v4 != 2 )
      return 0;
    return sub_72A2A0(a1 + 144, a2, a3, a4, 0);
  }
}
