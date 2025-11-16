// Function: sub_1688730
// Address: 0x1688730
//
__int64 __fastcall sub_1688730(__int64 a1, char *a2, int a3)
{
  __int64 result; // rax
  int v4; // ecx
  char v6; // di
  __int64 v7; // r10
  char v8; // dl

  result = *(unsigned int *)(a1 + 4);
  v4 = *(_DWORD *)(a1 + 8);
  v6 = *(_BYTE *)(a1 + 12);
  if ( a3 )
  {
    v7 = (__int64)&a2[a3 - 1 + 1];
    do
    {
      if ( --v4 )
      {
        result = (unsigned int)result >> 8;
      }
      else
      {
        v4 = 4;
        result = (unsigned int)(1103515245 * *(_DWORD *)a1 + 12345);
        *(_DWORD *)a1 = result;
      }
      v8 = *a2++;
      v6 ^= byte_42AE300[(unsigned __int8)(result ^ v8)];
      *(a2 - 1) = v6;
    }
    while ( (char *)v7 != a2 );
  }
  *(_DWORD *)(a1 + 4) = result;
  *(_DWORD *)(a1 + 8) = v4;
  *(_BYTE *)(a1 + 12) = v6;
  return result;
}
