// Function: sub_2F61B00
// Address: 0x2f61b00
//
__int64 __fastcall sub_2F61B00(__int64 *a1)
{
  __int64 v1; // r9
  unsigned __int64 v2; // r8
  unsigned __int64 v3; // r10
  __int64 v4; // rsi
  unsigned int v5; // edx
  __int64 result; // rax
  __int64 v7; // rax

  v1 = *a1;
  v2 = a1[1];
  v3 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  v4 = (*a1 >> 1) & 3;
  while ( 1 )
  {
    v5 = v4 | *(_DWORD *)(v3 + 24);
    result = *(_DWORD *)((*(a1 - 2) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(a1 - 2) >> 1) & 3;
    if ( v5 >= (unsigned int)result && (v5 > (unsigned int)result || v2 >= *(a1 - 1)) )
      break;
    v7 = *(a1 - 2);
    a1 -= 2;
    a1[2] = v7;
    a1[3] = a1[1];
  }
  *a1 = v1;
  a1[1] = v2;
  return result;
}
