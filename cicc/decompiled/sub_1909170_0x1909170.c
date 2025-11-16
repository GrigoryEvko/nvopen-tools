// Function: sub_1909170
// Address: 0x1909170
//
__int64 __fastcall sub_1909170(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r12d
  unsigned __int64 v5; // rcx
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdx
  unsigned int v8; // eax
  unsigned int v9; // eax

  v3 = 1 << (*(unsigned __int16 *)(a1 + 18) >> 1) >> 1;
  if ( !v3 )
    return 0;
  v5 = sub_1CCB4A0(a2, a3);
  v6 = v3;
  if ( v5 )
  {
    while ( 1 )
    {
      v7 = v6 % v5;
      v6 = v5;
      if ( !v7 )
        break;
      v5 = v7;
    }
    v3 = v5;
    if ( !(_DWORD)v5 )
      return 0;
  }
  if ( (v3 & (v3 - 1)) == 0 )
    return v3;
  v8 = v3 - 1;
  v9 = ((v8 | (v8 >> 1)) >> 2) | v8 | (v8 >> 1) | ((((v8 | (v8 >> 1)) >> 2) | v8 | (v8 >> 1)) >> 4);
  return ((v9 >> 8) | v9 | (((v9 >> 8) | v9) >> 16)) + 1;
}
