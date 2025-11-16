// Function: sub_8EEC80
// Address: 0x8eec80
//
void __fastcall sub_8EEC80(__int64 a1, unsigned __int64 a2)
{
  __int64 v2; // rdx
  int v3; // r9d
  __int64 v4; // rax
  unsigned __int64 v5; // rsi
  bool v6; // r8
  __int64 v7; // rax

  v2 = (unsigned int)a2;
  v3 = *(_DWORD *)(a1 + 2088);
  if ( (_DWORD)a2 && v3 > 0 )
  {
    v4 = 1;
    do
    {
      v5 = v2 + *(unsigned int *)(a1 + 4 * v4 + 4);
      v2 = 1;
      *(_DWORD *)(a1 + 4 * v4 + 4) = v5;
      a2 = HIDWORD(v5);
      v6 = v3 <= (int)v4++;
    }
    while ( !((unsigned __int8)a2 ^ 1 | v6) );
  }
  else
  {
    a2 = (unsigned int)a2;
  }
  if ( a2 )
  {
    v7 = v3++;
    *(_DWORD *)(a1 + 8 + 4 * v7) = a2;
  }
  *(_DWORD *)(a1 + 2088) = v3;
}
