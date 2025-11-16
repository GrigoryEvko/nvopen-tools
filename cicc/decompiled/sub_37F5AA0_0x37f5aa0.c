// Function: sub_37F5AA0
// Address: 0x37f5aa0
//
__int64 __fastcall sub_37F5AA0(__int64 a1, __int64 a2, unsigned int a3)
{
  int v4; // eax

  if ( !sub_37F5A90(a1, a2, a3) )
    return 0;
  v4 = sub_37F56A0(a1, a2, a3);
  return sub_37F5910(a1, *(_QWORD *)(a2 + 24), v4);
}
