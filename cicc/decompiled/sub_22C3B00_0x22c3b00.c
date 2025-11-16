// Function: sub_22C3B00
// Address: 0x22c3b00
//
__int64 __fastcall sub_22C3B00(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v4; // r9
  int v5; // edi
  __int64 v6; // rdx
  unsigned int v7; // eax
  __int64 v8; // rcx
  __int64 v9; // rsi
  __int64 result; // rax
  int v11; // r11d
  __int64 v12; // r10

  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v4 = a1 + 16;
    v5 = 1;
  }
  else
  {
    result = *(unsigned int *)(a1 + 24);
    v4 = *(_QWORD *)(a1 + 16);
    v5 = result - 1;
    if ( !(_DWORD)result )
    {
      *a3 = 0;
      return result;
    }
  }
  v6 = *(_QWORD *)(a2 + 16);
  v7 = v5 & (((unsigned int)v6 >> 4) ^ ((unsigned int)v6 >> 9));
  v8 = v4 + 24LL * v7;
  v9 = *(_QWORD *)(v8 + 16);
  if ( v6 == v9 )
  {
    *a3 = v8;
    return 1;
  }
  else
  {
    v11 = 1;
    v12 = 0;
    while ( v9 != -4096 )
    {
      if ( !v12 && v9 == -8192 )
        v12 = v8;
      v7 = v5 & (v11 + v7);
      v8 = v4 + 24LL * v7;
      v9 = *(_QWORD *)(v8 + 16);
      if ( v6 == v9 )
      {
        *a3 = v8;
        return 1;
      }
      ++v11;
    }
    if ( !v12 )
      v12 = v8;
    *a3 = v12;
    return 0;
  }
}
