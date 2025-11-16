// Function: sub_2ECEEC0
// Address: 0x2eceec0
//
__int64 __fastcall sub_2ECEEC0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rcx
  __int64 v3; // rdx
  unsigned int v4; // r12d
  unsigned int v5; // r13d
  __int64 v6; // rsi
  unsigned int v7; // edx

  result = *(_QWORD *)(a1 + 72);
  v2 = *(_QWORD *)(a1 + 64);
  if ( v2 == result )
    *(_DWORD *)(a1 + 172) = -1;
  v3 = *(_QWORD *)(a1 + 128);
  v4 = (*(_QWORD *)(a1 + 136) - v3) >> 3;
  if ( v4 )
  {
    v5 = 0;
    while ( 1 )
    {
      v6 = *(_QWORD *)(v3 + 8LL * v5);
      v7 = *(_DWORD *)(v6 + 236);
      if ( *(_DWORD *)(a1 + 24) == 1 )
        v7 = *(_DWORD *)(v6 + 232);
      if ( *(_DWORD *)(a1 + 172) > v7 )
        *(_DWORD *)(a1 + 172) = v7;
      result = (result - v2) >> 3;
      if ( (unsigned int)qword_5021648 <= (unsigned int)result )
        break;
      sub_2ECEBE0(a1, v6, v7, 1, v5);
      v3 = *(_QWORD *)(a1 + 128);
      result = (*(_QWORD *)(a1 + 136) - v3) >> 3;
      if ( v4 == (_DWORD)result )
      {
        if ( v4 <= ++v5 )
          break;
      }
      else if ( --v4 <= v5 )
      {
        break;
      }
      result = *(_QWORD *)(a1 + 72);
      v2 = *(_QWORD *)(a1 + 64);
    }
  }
  *(_BYTE *)(a1 + 160) = 0;
  return result;
}
