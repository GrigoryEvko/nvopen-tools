// Function: sub_2588040
// Address: 0x2588040
//
__int64 __fastcall sub_2588040(__int64 a1, __int64 a2, __int64 *a3, int a4, bool *a5, __int64 a6, __int64 *a7)
{
  unsigned int v9; // eax
  unsigned int v10; // r12d
  __int64 v12; // rax

  *a5 = 0;
  LOBYTE(v9) = sub_2553E90(a1, (unsigned __int64)a3);
  v10 = v9;
  if ( (_BYTE)v9 )
  {
    *a5 = 1;
  }
  else if ( a2 )
  {
    v12 = sub_2527F10(a1, *a3, a3[1], a2, a4, 0, 1);
    if ( a7 )
      *a7 = v12;
    if ( v12 && (*(_WORD *)(v12 + 98) & 7) == 7 )
    {
      v10 = 1;
      *a5 = (*(_WORD *)(v12 + 96) & 7) == 7;
    }
  }
  return v10;
}
