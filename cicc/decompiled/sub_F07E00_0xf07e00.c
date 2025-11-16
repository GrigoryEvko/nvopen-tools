// Function: sub_F07E00
// Address: 0xf07e00
//
bool __fastcall sub_F07E00(_BYTE *a1, unsigned int *a2, _QWORD *a3)
{
  unsigned int v3; // r12d
  __int64 v4; // rax
  unsigned int v5; // ebx
  __int64 v6; // r8
  int v7; // eax
  unsigned __int64 v9; // rax
  unsigned int v10; // r13d
  __int64 v11; // rdx
  unsigned __int64 v12; // rax
  unsigned int v13; // r13d
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rax

  v3 = *a2;
  v4 = *(_QWORD *)(*(_QWORD *)(*a3 - 8LL) + 32LL * (2 * (unsigned int)a3[1] + 2));
  v5 = *(_DWORD *)(v4 + 32);
  v6 = v4 + 24;
  if ( *a1 )
  {
    if ( v5 <= 0x40 )
    {
      v14 = *(_QWORD *)(v4 + 24);
      v7 = *(_DWORD *)(v4 + 32);
      if ( v14 )
      {
        _BitScanReverse64(&v15, v14);
        v7 = v5 - 64 + (v15 ^ 0x3F);
      }
    }
    else
    {
      v7 = sub_C444A0(v4 + 24);
    }
    return v5 - v7 <= v3;
  }
  else
  {
    v9 = *(_QWORD *)(v4 + 24);
    v10 = v5 + 1;
    v11 = 1LL << ((unsigned __int8)v5 - 1);
    if ( v5 > 0x40 )
    {
      if ( (*(_QWORD *)(v9 + 8LL * ((v5 - 1) >> 6)) & v11) != 0 )
        v13 = v10 - sub_C44500(v6);
      else
        v13 = v10 - sub_C444A0(v6);
    }
    else if ( (v11 & v9) != 0 )
    {
      if ( v5 )
      {
        v12 = ~(v9 << (64 - (unsigned __int8)v5));
        if ( v12 )
        {
          _BitScanReverse64(&v12, v12);
          v13 = v10 - (v12 ^ 0x3F);
        }
        else
        {
          v13 = v5 - 63;
        }
      }
      else
      {
        v13 = 1;
      }
    }
    else
    {
      v13 = 1;
      if ( v9 )
      {
        _BitScanReverse64(&v9, v9);
        v13 = 65 - (v9 ^ 0x3F);
      }
    }
    return v13 <= v3;
  }
}
