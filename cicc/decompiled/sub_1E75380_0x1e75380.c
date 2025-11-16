// Function: sub_1E75380
// Address: 0x1e75380
//
__int64 __fastcall sub_1E75380(__int64 a1)
{
  __int64 v1; // rdx
  __int64 result; // rax
  int v3; // r13d
  unsigned int v4; // r15d
  _BYTE *v5; // rsi
  __int64 v6; // r12
  _DWORD *v7; // r14
  int v8; // [rsp+Ch] [rbp-44h]
  _DWORD *v9; // [rsp+18h] [rbp-38h] BYREF

  if ( *(_QWORD *)(a1 + 64) == *(_QWORD *)(a1 + 72) )
    *(_DWORD *)(a1 + 172) = -1;
  v1 = *(_QWORD *)(a1 + 128);
  v8 = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 4LL);
  result = (*(_QWORD *)(a1 + 136) - v1) >> 3;
  v3 = result;
  if ( (_DWORD)result )
  {
    v4 = 0;
    while ( 1 )
    {
      v7 = *(_DWORD **)(v1 + 8LL * v4);
      result = (unsigned int)v7[63];
      if ( *(_DWORD *)(a1 + 24) == 1 )
        result = (unsigned int)v7[62];
      if ( *(_DWORD *)(a1 + 172) > (unsigned int)result )
        *(_DWORD *)(a1 + 172) = result;
      if ( (v8 || *(_DWORD *)(a1 + 164) >= (unsigned int)result)
        && (result = sub_1E72C10(a1, (__int64)v7), !(_BYTE)result) )
      {
        v5 = *(_BYTE **)(a1 + 72);
        result = (__int64)&v5[-*(_QWORD *)(a1 + 64)] >> 3;
        if ( dword_4FC7CA0 <= (unsigned int)result )
          break;
        v9 = v7;
        if ( v5 == *(_BYTE **)(a1 + 80) )
        {
          sub_1CFD630(a1 + 64, v5, &v9);
          v7 = v9;
        }
        else
        {
          if ( v5 )
          {
            *(_QWORD *)v5 = v7;
            v5 = *(_BYTE **)(a1 + 72);
          }
          *(_QWORD *)(a1 + 72) = v5 + 8;
        }
        v7[49] |= *(_DWORD *)(a1 + 24);
        --v3;
        v6 = *(_QWORD *)(a1 + 128) + 8LL * v4;
        *(_DWORD *)(*(_QWORD *)v6 + 196LL) &= ~*(_DWORD *)(a1 + 88);
        result = *(_QWORD *)(*(_QWORD *)(a1 + 136) - 8LL);
        *(_QWORD *)v6 = result;
        *(_QWORD *)(a1 + 136) -= 8LL;
        if ( v3 == v4 )
          break;
      }
      else if ( v3 == ++v4 )
      {
        break;
      }
      v1 = *(_QWORD *)(a1 + 128);
    }
  }
  *(_BYTE *)(a1 + 160) = 0;
  return result;
}
