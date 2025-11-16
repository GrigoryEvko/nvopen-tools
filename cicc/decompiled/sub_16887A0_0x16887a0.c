// Function: sub_16887A0
// Address: 0x16887a0
//
__int64 __fastcall sub_16887A0(__int64 a1, _BYTE *a2, int a3)
{
  int v4; // ecx
  unsigned int v5; // edx
  __int64 result; // rax
  __int64 v7; // r10
  unsigned __int8 v8; // al
  char v9; // r8

  v4 = *(_DWORD *)(a1 + 8);
  v5 = *(_DWORD *)(a1 + 4);
  result = *(unsigned __int8 *)(a1 + 12);
  if ( a3 )
  {
    v7 = (__int64)&a2[a3 - 1 + 1];
    do
    {
      if ( --v4 )
      {
        v5 >>= 8;
      }
      else
      {
        v4 = 4;
        v5 = 1103515245 * *(_DWORD *)a1 + 12345;
        *(_DWORD *)a1 = v5;
      }
      v8 = *a2++ ^ result;
      v9 = byte_42AE200[v8];
      result = (unsigned __int8)*(a2 - 1);
      *(a2 - 1) = v5 ^ v9;
    }
    while ( (_BYTE *)v7 != a2 );
  }
  *(_DWORD *)(a1 + 4) = v5;
  *(_DWORD *)(a1 + 8) = v4;
  *(_BYTE *)(a1 + 12) = result;
  return result;
}
