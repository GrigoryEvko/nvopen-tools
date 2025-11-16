// Function: sub_8EE5A0
// Address: 0x8ee5a0
//
__int64 __fastcall sub_8EE5A0(__int64 a1, int a2)
{
  int v2; // eax
  unsigned int v3; // r8d
  __int64 v4; // rax
  bool v5; // dl
  bool v6; // cl

  v2 = a2 / 8;
  v3 = a2 / 8;
  if ( (*(unsigned __int8 *)(a1 + a2 / 8) & ~(-1 << (a2 % 8))) != 0 )
    return 1;
  if ( v2 )
  {
    v4 = v2 - 1;
    do
    {
      v5 = *(_BYTE *)(a1 + v4) != 0;
      v6 = (_DWORD)v4-- == 0;
    }
    while ( !v5 && !v6 );
    return v5;
  }
  return v3;
}
