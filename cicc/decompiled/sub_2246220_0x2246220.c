// Function: sub_2246220
// Address: 0x2246220
//
__int64 __fastcall sub_2246220(_DWORD *a1, unsigned __int64 a2, __int64 a3, __int16 a4, char a5)
{
  _DWORD *v5; // rcx
  unsigned __int64 v6; // rax
  bool v8; // cf
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax

  if ( a5 )
  {
    v5 = a1;
    do
    {
      *--v5 = *(_DWORD *)(a3 + 4 * (a2 % 0xA) + 16);
      v6 = a2;
      a2 /= 0xAu;
    }
    while ( v6 > 9 );
  }
  else if ( (a4 & 0x4A) == 0x40 )
  {
    v5 = a1;
    do
    {
      --v5;
      v10 = a2 & 7;
      a2 >>= 3;
      *v5 = *(_DWORD *)(a3 + 4 * v10 + 16);
    }
    while ( a2 );
  }
  else
  {
    v8 = (a4 & 0x4000) == 0;
    v5 = a1;
    do
    {
      --v5;
      v9 = (-(__int64)v8 & 0xFFFFFFFFFFFFFFF0LL) + 20 + (a2 & 0xF);
      a2 >>= 4;
      *v5 = *(_DWORD *)(a3 + 4 * v9);
    }
    while ( a2 );
  }
  return a1 - v5;
}
