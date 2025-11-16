// Function: sub_2FB0100
// Address: 0x2fb0100
//
__int64 __fastcall sub_2FB0100(__int64 a1)
{
  __int64 *v1; // r13
  int v2; // r11d
  __int64 v3; // rdx
  unsigned int v4; // r8d
  __int64 v5; // r10
  unsigned int i; // ecx
  unsigned int v10; // ecx
  unsigned int v11; // r9d
  unsigned int v12; // esi
  __int64 v13; // r10
  unsigned __int64 v14; // r8
  __int64 v15; // rax
  unsigned __int64 v16; // r8
  unsigned __int8 v19; // [rsp+1h] [rbp-29h]

  v1 = *(__int64 **)(a1 + 32);
  v2 = *((_DWORD *)v1 + 16);
  if ( v2 )
  {
    v3 = 0;
    v4 = (unsigned int)(v2 - 1) >> 6;
    v5 = *v1;
    while ( 1 )
    {
      _RSI = *(_QWORD *)(v5 + 8 * v3);
      if ( v4 == (_DWORD)v3 )
        _RSI = (0xFFFFFFFFFFFFFFFFLL >> -(char)v2) & *(_QWORD *)(v5 + 8 * v3);
      if ( _RSI )
        break;
      if ( v4 + 1 == ++v3 )
        goto LABEL_7;
    }
    __asm { tzcnt   rsi, rsi }
    v19 = 1;
    for ( i = ((_DWORD)v3 << 6) + _RSI; i != -1; i = ((_DWORD)v15 << 6) + _RCX )
    {
      if ( *(int *)(*(_QWORD *)(a1 + 24) + 112LL * i + 16) <= 0 )
      {
        v19 = 0;
        *(_QWORD *)(**(_QWORD **)(a1 + 32) + 8LL * (i >> 6)) &= ~(1LL << i);
        v2 = *((_DWORD *)v1 + 16);
      }
      v10 = i + 1;
      if ( v2 == v10 )
        break;
      v11 = v10 >> 6;
      v12 = (unsigned int)(v2 - 1) >> 6;
      if ( v10 >> 6 > v12 )
        break;
      v13 = *v1;
      v14 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (v10 & 0x3F));
      if ( (v10 & 0x3F) == 0 )
        v14 = 0;
      v15 = v11;
      v16 = ~v14;
      while ( 1 )
      {
        _RCX = *(_QWORD *)(v13 + 8 * v15);
        if ( v11 == (_DWORD)v15 )
          _RCX = v16 & *(_QWORD *)(v13 + 8 * v15);
        if ( v12 == (_DWORD)v15 )
          _RCX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v2;
        if ( _RCX )
          break;
        if ( v12 < (unsigned int)++v15 )
          goto LABEL_8;
      }
      __asm { tzcnt   rcx, rcx }
    }
  }
  else
  {
LABEL_7:
    v19 = 1;
  }
LABEL_8:
  *(_QWORD *)(a1 + 32) = 0;
  return v19;
}
