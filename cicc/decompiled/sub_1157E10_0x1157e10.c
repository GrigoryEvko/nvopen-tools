// Function: sub_1157E10
// Address: 0x1157e10
//
__int64 __fastcall sub_1157E10(__int64 **a1, __int64 a2)
{
  unsigned int v4; // r13d
  __int64 v5; // rdx
  int v7; // r14d
  unsigned __int64 v8; // r14
  bool v10; // al
  __int64 result; // rax
  __int64 v12; // r13
  __int64 v13; // rdx
  _BYTE *v14; // rax
  unsigned int v15; // r14d
  __int64 v17; // rsi
  int v18; // r15d
  unsigned __int64 v19; // rax
  int v21; // r13d
  char v22; // r15
  unsigned int v23; // r14d
  __int64 v24; // rax
  unsigned int v25; // r8d
  __int64 v27; // rdi
  int v28; // ecx
  unsigned __int64 v29; // rax
  __int64 v31; // r15
  __int64 v32; // r13
  unsigned int v33; // [rsp+8h] [rbp-38h]
  int v34; // [rsp+Ch] [rbp-34h]

  if ( *(_BYTE *)a2 == 17 )
  {
    v4 = *(_DWORD *)(a2 + 32);
    v5 = 1LL << ((unsigned __int8)v4 - 1);
    _RAX = *(_QWORD *)(a2 + 24);
    if ( v4 > 0x40 )
    {
      if ( (*(_QWORD *)(_RAX + 8LL * ((v4 - 1) >> 6)) & v5) != 0 )
      {
        v7 = sub_C44500(a2 + 24);
        LODWORD(_RAX) = sub_C44590(a2 + 24);
LABEL_12:
        v10 = v7 + (_DWORD)_RAX == v4;
        goto LABEL_13;
      }
    }
    else if ( (v5 & _RAX) != 0 )
    {
      if ( v4 )
      {
        v7 = 64;
        if ( _RAX << (64 - (unsigned __int8)v4) != -1 )
        {
          _BitScanReverse64(&v8, ~(_RAX << (64 - (unsigned __int8)v4)));
          v7 = v8 ^ 0x3F;
        }
      }
      else
      {
        v7 = 0;
      }
      __asm { tzcnt   rax, rax }
      if ( (unsigned int)_RAX > v4 )
        LODWORD(_RAX) = *(_DWORD *)(a2 + 32);
      goto LABEL_12;
    }
    return 0;
  }
  v12 = *(_QWORD *)(a2 + 8);
  v13 = (unsigned int)*(unsigned __int8 *)(v12 + 8) - 17;
  if ( (unsigned int)v13 > 1 || *(_BYTE *)a2 > 0x15u )
    return 0;
  v14 = sub_AD7630(a2, 0, v13);
  if ( !v14 || *v14 != 17 )
  {
    if ( *(_BYTE *)(v12 + 8) == 17 )
    {
      v21 = *(_DWORD *)(v12 + 32);
      if ( v21 )
      {
        v22 = 0;
        v23 = 0;
        while ( 1 )
        {
          v24 = sub_AD69F0((unsigned __int8 *)a2, v23);
          if ( !v24 )
            break;
          if ( *(_BYTE *)v24 != 13 )
          {
            if ( *(_BYTE *)v24 != 17 )
              return 0;
            v25 = *(_DWORD *)(v24 + 32);
            _RSI = *(_QWORD *)(v24 + 24);
            v27 = 1LL << ((unsigned __int8)v25 - 1);
            if ( v25 > 0x40 )
            {
              if ( (*(_QWORD *)(_RSI + 8LL * ((v25 - 1) >> 6)) & v27) == 0 )
                return 0;
              v31 = v24 + 24;
              v33 = *(_DWORD *)(v24 + 32);
              v34 = sub_C44500(v24 + 24);
              LODWORD(_RAX) = sub_C44590(v31);
              v25 = v33;
              v28 = v34;
            }
            else
            {
              if ( (v27 & _RSI) == 0 )
                return 0;
              if ( v25 )
              {
                v28 = 64;
                if ( _RSI << (64 - (unsigned __int8)v25) != -1 )
                {
                  _BitScanReverse64(&v29, ~(_RSI << (64 - (unsigned __int8)v25)));
                  v28 = v29 ^ 0x3F;
                }
              }
              else
              {
                v28 = 0;
              }
              __asm { tzcnt   rax, rsi }
              if ( (unsigned int)_RAX > v25 )
                LODWORD(_RAX) = v25;
            }
            if ( v25 != v28 + (_DWORD)_RAX )
              return 0;
            v22 = 1;
          }
          if ( v21 == ++v23 )
          {
            if ( v22 )
              goto LABEL_14;
            return 0;
          }
        }
      }
    }
    return 0;
  }
  v15 = *((_DWORD *)v14 + 8);
  _RDX = *((_QWORD *)v14 + 3);
  v17 = 1LL << ((unsigned __int8)v15 - 1);
  if ( v15 > 0x40 )
  {
    if ( (*(_QWORD *)(_RDX + 8LL * ((v15 - 1) >> 6)) & v17) == 0 )
      return 0;
    v32 = (__int64)(v14 + 24);
    v18 = sub_C44500((__int64)(v14 + 24));
    LODWORD(_RAX) = sub_C44590(v32);
  }
  else
  {
    if ( (v17 & _RDX) == 0 )
      return 0;
    if ( v15 )
    {
      v18 = 64;
      if ( _RDX << (64 - (unsigned __int8)v15) != -1 )
      {
        _BitScanReverse64(&v19, ~(_RDX << (64 - (unsigned __int8)v15)));
        v18 = v19 ^ 0x3F;
      }
    }
    else
    {
      v18 = 0;
    }
    __asm { tzcnt   rax, rdx }
    if ( (unsigned int)_RAX > v15 )
      LODWORD(_RAX) = v15;
  }
  v10 = v18 + (_DWORD)_RAX == v15;
LABEL_13:
  if ( !v10 )
    return 0;
LABEL_14:
  result = 1;
  if ( *a1 )
    **a1 = a2;
  return result;
}
