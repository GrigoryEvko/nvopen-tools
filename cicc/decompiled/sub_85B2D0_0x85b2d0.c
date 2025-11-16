// Function: sub_85B2D0
// Address: 0x85b2d0
//
__int64 __fastcall sub_85B2D0(__int64 a1, unsigned int a2, char a3)
{
  FILE *v3; // r14

  v3 = (FILE *)(a1 + 48);
  if ( a3 == 3 )
    return sub_6853B0(3u, a2, (FILE *)(a1 + 48), a1);
  if ( a3 == 4 || !dword_4F04C64 && sub_729F20(*(_DWORD *)(a1 + 48)) )
    return sub_685460(a2, v3, a1);
  return sub_685490(a2, v3, a1);
}
