// Function: sub_22918F0
// Address: 0x22918f0
//
__int64 __fastcall sub_22918F0(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 *a6,
        _DWORD *a7,
        __int64 a8)
{
  __int64 v10; // r12
  int v11; // ebx
  unsigned int v12; // edx
  unsigned __int64 v13; // r8
  char v14; // al
  unsigned int v15; // r10d
  char v16; // al
  unsigned int v17; // r10d
  char v18; // al
  unsigned int v19; // r10d
  unsigned int v20; // edx
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rsi
  unsigned int v24; // edx
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rsi
  unsigned int v27; // [rsp+4h] [rbp-3Ch]
  unsigned int v28; // [rsp+4h] [rbp-3Ch]

  v10 = a2;
  v11 = (int)a6;
  v12 = *(_DWORD *)(a1 + 32);
  if ( (unsigned int)qword_4FDB3E8 >= v12 )
  {
    while ( v12 >= (unsigned int)v10 )
    {
      v13 = *a6;
      if ( (*a6 & 1) != 0 )
      {
        if ( ((((v13 >> 1) & ~(-1LL << (*a6 >> 58))) >> v10) & 1) != 0 )
          goto LABEL_8;
      }
      else if ( ((*(_QWORD *)(*(_QWORD *)v13 + 8LL * ((unsigned int)v10 >> 6)) >> v10) & 1) != 0 )
      {
LABEL_8:
        if ( *a7 < (unsigned int)v10 )
        {
          *a7 = v10;
          sub_2291120(a1, a3, a4, a5, v10);
          sub_2291360(a1, a3, a4, a5, v10);
          sub_2290FB0(a1, a3, a4, a5, v10);
        }
        v14 = sub_2291860(a1, 1, v10, a5, a8);
        v15 = 0;
        if ( v14 )
          v15 = sub_22918F0(a1, (int)v10 + 1, a3, a4, a5, v11, (__int64)a7, a8);
        v27 = v15;
        v16 = sub_2291860(a1, 2, v10, a5, a8);
        v17 = v27;
        if ( v16 )
          v17 = sub_22918F0(a1, (int)v10 + 1, a3, a4, a5, v11, (__int64)a7, a8) + v27;
        v28 = v17;
        v18 = sub_2291860(a1, 4, v10, a5, a8);
        v19 = v28;
        if ( v18 )
          v19 = sub_22918F0(a1, (int)v10 + 1, a3, a4, a5, v11, (__int64)a7, a8) + v28;
        *(_BYTE *)(a5 + 144 * v10 + 136) = 7;
        return v19;
      }
      v10 = (unsigned int)(v10 + 1);
    }
    if ( v12 )
    {
      v24 = 1;
      do
      {
        v26 = *a6;
        if ( (*a6 & 1) != 0 )
          v25 = (((v26 >> 1) & ~(-1LL << (*a6 >> 58))) >> v24) & 1;
        else
          v25 = (*(_QWORD *)(*(_QWORD *)v26 + 8LL * (v24 >> 6)) >> v24) & 1LL;
        if ( (_BYTE)v25 )
          *(_BYTE *)(a5 + 144LL * v24 + 137) |= *(_BYTE *)(a5 + 144LL * v24 + 136);
        ++v24;
      }
      while ( *(_DWORD *)(a1 + 32) >= v24 );
    }
  }
  else if ( v12 )
  {
    v20 = 1;
    do
    {
      v22 = *a6;
      if ( (*a6 & 1) != 0 )
        v21 = (((v22 >> 1) & ~(-1LL << (*a6 >> 58))) >> v20) & 1;
      else
        v21 = (*(_QWORD *)(*(_QWORD *)v22 + 8LL * (v20 >> 6)) >> v20) & 1LL;
      if ( (_BYTE)v21 )
        *(_BYTE *)(a5 + 144LL * v20 + 137) = 7;
      ++v20;
    }
    while ( *(_DWORD *)(a1 + 32) >= v20 );
  }
  return 1;
}
