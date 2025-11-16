// Function: sub_1EEA3B0
// Address: 0x1eea3b0
//
__int64 __fastcall sub_1EEA3B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rcx
  __int64 v6; // rax
  __int64 i; // rdx
  __int64 result; // rax
  _QWORD *v9; // rcx
  int v10; // r8d
  int v11; // r9d
  unsigned int v12; // edx
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned int v17; // esi
  bool v18; // zf
  __int64 v19; // rsi
  __int64 v20; // rax

  if ( *(_BYTE *)(a1 + 44) )
  {
    v20 = *(_QWORD *)(a1 + 32);
    if ( !v20 )
      BUG();
    if ( (*(_BYTE *)v20 & 4) == 0 && (*(_BYTE *)(v20 + 46) & 8) != 0 )
    {
      do
        v20 = *(_QWORD *)(v20 + 8);
      while ( (*(_BYTE *)(v20 + 46) & 8) != 0 );
    }
    v5 = *(_QWORD *)(v20 + 8);
    *(_QWORD *)(a1 + 32) = v5;
  }
  else
  {
    v5 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 32LL);
    *(_BYTE *)(a1 + 44) = 1;
    *(_QWORD *)(a1 + 32) = v5;
  }
  v6 = *(_QWORD *)(a1 + 48);
  for ( i = v6 + 16LL * *(unsigned int *)(a1 + 56); v6 != i; v6 += 16 )
  {
    if ( *(_QWORD *)(v6 + 8) == v5 )
    {
      *(_DWORD *)(v6 + 4) = 0;
      *(_QWORD *)(v6 + 8) = 0;
    }
  }
  result = (unsigned int)**(unsigned __int16 **)(v5 + 16) - 12;
  if ( (unsigned __int16)(**(_WORD **)(v5 + 16) - 12) > 1u )
  {
    sub_1EEA150((__int64 *)a1, a2, i, v5, a5);
    v12 = *(_DWORD *)(a1 + 120);
    v13 = (unsigned int)(*(_DWORD *)(a1 + 144) + 63) >> 6;
    if ( (unsigned int)v13 > (v12 + 63) >> 6 )
      v13 = (v12 + 63) >> 6;
    if ( (_DWORD)v13 )
    {
      v14 = 8 * v13;
      v15 = 0;
      do
      {
        v16 = *(_QWORD *)(*(_QWORD *)(a1 + 128) + v15);
        v9 = (_QWORD *)(v15 + *(_QWORD *)(a1 + 104));
        v15 += 8;
        *v9 &= ~v16;
      }
      while ( v14 != v15 );
      v12 = *(_DWORD *)(a1 + 120);
    }
    v17 = *(_DWORD *)(a1 + 168);
    if ( v17 > v12 )
    {
      sub_13A49F0(a1 + 104, v17, 0, (int)v9, v10, v11);
      v17 = *(_DWORD *)(a1 + 168);
    }
    v18 = (v17 + 63) >> 6 == 0;
    result = (v17 + 63) >> 6;
    v19 = result;
    if ( !v18 )
    {
      result = 0;
      do
      {
        *(_QWORD *)(*(_QWORD *)(a1 + 104) + 8 * result) |= *(_QWORD *)(*(_QWORD *)(a1 + 152) + 8 * result);
        ++result;
      }
      while ( v19 != result );
    }
  }
  return result;
}
