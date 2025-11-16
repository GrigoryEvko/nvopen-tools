// Function: sub_1DA06D0
// Address: 0x1da06d0
//
unsigned __int64 __fastcall sub_1DA06D0(__int64 a1)
{
  int v1; // ecx
  unsigned __int64 v2; // rax
  unsigned __int64 result; // rax
  bool v4; // zf
  int v5; // esi
  __int64 *v6; // rdx
  unsigned int v7; // r8d
  __int64 v10; // rax
  __int64 v11; // rdx
  int v12; // ecx
  int v13; // eax
  int v16; // ecx
  unsigned int v17; // eax
  __int64 v18; // rax
  unsigned int v20; // ecx
  __int64 v21; // rax

  v1 = *(_DWORD *)(a1 + 24) + 1;
  v2 = *(_QWORD *)(a1 + 32);
  *(_DWORD *)(a1 + 24) = v1;
  result = v2 >> 1;
  v4 = *(_BYTE *)a1 == 0;
  *(_QWORD *)(a1 + 32) = result;
  if ( !v4 )
    return result;
  if ( !result )
  {
LABEL_7:
    v6 = *(__int64 **)(a1 + 16);
    v7 = v1 & 0x7F;
    if ( v6[(v7 >> 6) + 3] & (-1LL << v1) )
    {
      __asm { tzcnt   rax, rax }
      v20 = _RAX + (v1 & 0x40);
    }
    else
    {
      if ( (unsigned __int8)(v1 & 0x7F) >> 6 == 1 || (_RAX = v6[4]) == 0 )
      {
LABEL_10:
        v10 = *(_QWORD *)(a1 + 8);
        v11 = *v6;
        *(_DWORD *)(a1 + 28) = 0;
        result = v10 + 8;
        *(_QWORD *)(a1 + 16) = v11;
        if ( v11 == result )
        {
          *(_BYTE *)a1 = 1;
        }
        else
        {
          v12 = 0;
          v13 = *(_DWORD *)(v11 + 16) << 7;
          *(_DWORD *)(a1 + 24) = v13;
          _RSI = *(_QWORD *)(v11 + 24);
          if ( !_RSI )
          {
            _RSI = *(_QWORD *)(v11 + 32);
            v12 = 64;
            if ( !_RSI )
            {
              _RSI = *(_QWORD *)(v11 + 40);
              v12 = 128;
            }
          }
          __asm { tzcnt   rsi, rsi }
          v16 = _RSI + v12;
          v17 = v16 + v13;
          *(_DWORD *)(a1 + 24) = v17;
          v18 = (v17 >> 6) & 1;
          *(_DWORD *)(a1 + 28) = v18;
          result = *(_QWORD *)(v11 + 8 * v18 + 24) >> v16;
          *(_QWORD *)(a1 + 32) = result;
        }
        return result;
      }
      __asm { tzcnt   rax, rax }
      v20 = _RAX + 64;
    }
    if ( v7 )
    {
      v21 = v20 >> 6;
      *(_DWORD *)(a1 + 28) = v21;
      *(_QWORD *)(a1 + 32) = (unsigned __int64)v6[v21 + 3] >> v20;
      result = (unsigned int)(*((_DWORD *)v6 + 4) << 7);
      *(_DWORD *)(a1 + 24) = result + v20;
      return result;
    }
    goto LABEL_10;
  }
  while ( (result & 1) == 0 )
  {
    v5 = *(_DWORD *)(a1 + 24);
    result >>= 1;
    *(_QWORD *)(a1 + 32) = result;
    LOBYTE(v1) = v5 + 1;
    *(_DWORD *)(a1 + 24) = v5 + 1;
    if ( !result )
      goto LABEL_7;
  }
  return result;
}
