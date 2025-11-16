// Function: sub_130DC30
// Address: 0x130dc30
//
unsigned __int64 __fastcall sub_130DC30(__int64 a1, int a2, int a3, int a4, int a5)
{
  char v6; // r14
  int v7; // r12d
  unsigned __int64 result; // rax
  unsigned __int64 v10; // r13
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rcx
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // r8
  unsigned __int64 v16; // rsi

  v6 = a3;
  v7 = a4;
  *(_DWORD *)a1 = a2;
  *(_DWORD *)(a1 + 4) = a3;
  *(_DWORD *)(a1 + 8) = a4;
  *(_DWORD *)(a1 + 12) = a5;
  result = sub_130DC10(a3, a4, a5);
  *(_BYTE *)(a1 + 16) = (result & 0xFFF) == 0;
  if ( result <= 0x3FFF )
  {
    *(_BYTE *)(a1 + 17) = 1;
    v10 = result;
    v11 = sub_130DC10(v6, v7, a5);
    v12 = 4096;
    v13 = v11;
    v14 = 0x1000 / v11;
    do
    {
      v15 = v12;
      v12 += 4096LL;
      v16 = v13 * v14;
      v14 = v12 / v13;
    }
    while ( v16 != v15 );
    result = 0;
    *(_DWORD *)(a1 + 20) = v16 >> 12;
    if ( v10 >= 0x1001 )
      v7 = 0;
  }
  else
  {
    *(_BYTE *)(a1 + 17) = 0;
    v7 = 0;
    *(_DWORD *)(a1 + 20) = 0;
  }
  *(_DWORD *)(a1 + 24) = v7;
  return result;
}
