// Function: sub_131C5B0
// Address: 0x131c5b0
//
__int64 __fastcall sub_131C5B0(__int64 a1, unsigned __int64 a2, unsigned __int64 a3, __int64 a4)
{
  int v5; // eax
  unsigned int v6; // r8d
  char v7; // cl
  __int64 v8; // rdx
  int v9; // esi
  unsigned int v10; // esi
  __int64 v11; // rdx
  char v12; // cl
  __int64 v13; // r8
  int v14; // edx
  unsigned int v15; // edx
  int v16; // ecx
  _DWORD *v17; // rax
  unsigned __int64 v18; // rdx
  unsigned __int64 v20; // rcx
  int v21; // esi
  unsigned __int64 v22; // rcx

  v5 = a4;
  v6 = 1;
  if ( (unsigned __int64)(a4 - 1) <= 0x3F )
  {
    if ( a2 > 0x3800 )
      return 0;
    if ( a2 )
    {
      if ( a2 > 8 )
      {
        v7 = 7;
        _BitScanReverse64((unsigned __int64 *)&v8, 2 * a2 - 1);
        if ( (unsigned int)v8 >= 7 )
          v7 = v8;
        v9 = (((-1LL << (v7 - 3)) & (a2 - 1)) >> (v7 - 3)) & 3;
        if ( (unsigned int)v8 < 6 )
          LODWORD(v8) = 6;
        v10 = v9 + 4 * v8 - 23;
        if ( !a3 )
          return 0;
LABEL_10:
        v11 = 14336;
        if ( a3 <= 0x3800 )
          v11 = a3;
        if ( a3 <= 8 )
        {
          if ( a3 == 1 )
          {
            v15 = 0;
          }
          else
          {
            _BitScanReverse64(&v22, v11 - 1);
            _BitScanReverse64(&v22, 1LL << ((unsigned __int8)v22 + 1));
            v15 = v22 - 3;
            if ( (int)v22 <= 2 )
              v15 = 0;
          }
        }
        else
        {
          v12 = 7;
          _BitScanReverse64((unsigned __int64 *)&v13, 2 * v11 - 1);
          if ( (unsigned int)v13 >= 7 )
            v12 = v13;
          v14 = (((-1LL << (v12 - 3)) & (unsigned __int64)(v11 - 1)) >> (v12 - 3)) & 3;
          if ( (unsigned int)v13 < 6 )
            LODWORD(v13) = 6;
          v15 = v14 + 4 * v13 - 23;
        }
LABEL_18:
        if ( v15 >= v10 )
          goto LABEL_19;
        return 0;
      }
      if ( a2 != 1 )
      {
        _BitScanReverse64(&v20, a2 - 1);
        _BitScanReverse64(&a2, 1LL << ((unsigned __int8)v20 + 1));
        if ( v21 > 2 )
        {
          v10 = a2 - 3;
          v15 = 0;
          if ( !a3 )
            goto LABEL_18;
          goto LABEL_10;
        }
      }
    }
    v10 = 0;
    if ( !a3 )
    {
      v15 = 0;
LABEL_19:
      v16 = v5;
      v17 = (_DWORD *)(a1 + 4LL * v10);
      v18 = a1 + 4 * (v10 + (unsigned __int64)(v15 - v10)) + 4;
      do
        *v17++ = v16;
      while ( (_DWORD *)v18 != v17 );
      return 0;
    }
    goto LABEL_10;
  }
  return v6;
}
