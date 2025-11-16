// Function: sub_1E17F60
// Address: 0x1e17f60
//
__int64 __fastcall sub_1E17F60(__int64 a1, unsigned int a2, unsigned __int64 *a3, __int64 a4)
{
  __int64 v5; // r13
  __int64 v8; // r15
  int v9; // eax
  __int64 v10; // rcx
  bool v11; // dl
  unsigned __int8 v12; // si
  __int64 v13; // rax
  _BYTE *v14; // rax
  char v15; // dl
  unsigned __int64 v16; // rsi
  char v17; // dl
  unsigned __int64 v18; // rax
  int v19; // ecx
  bool v20; // r9
  __int64 v21; // r8
  int v22; // eax
  bool v23; // dl
  __int64 v24; // rcx
  __int64 v25; // rax
  _BYTE *v26; // rax
  __int64 v27; // rcx
  _BYTE *v28; // rcx
  __int64 v29; // r10

  v5 = *(_QWORD *)(a1 + 32) + 40LL * a2;
  if ( *(_BYTE *)v5 )
    return 0;
  v8 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) & 1LL;
  if ( (*(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) & 1) != 0 || (unsigned int)sub_1E16360(a1) <= a2 )
  {
    v9 = *(_DWORD *)(v5 + 8);
    if ( v9 >= 0 || (v13 = v9 & 0x7FFFFFFF, (unsigned int)v13 >= *(_DWORD *)(a4 + 336)) )
    {
      v10 = 0;
      v11 = 0;
      v12 = 0;
    }
    else
    {
      v14 = (_BYTE *)(*(_QWORD *)(a4 + 328) + 8 * v13);
      v12 = *v14 & 1;
      v11 = (*v14 & 2) != 0;
      v10 = *(_QWORD *)v14 >> 2;
    }
    return (4 * v10) | v12 | (2LL * v11);
  }
  else
  {
    v15 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 40LL) + 8LL * a2 + 3);
    if ( (unsigned __int8)(v15 - 6) > 5u )
    {
      v22 = *(_DWORD *)(v5 + 8);
      v23 = 0;
      v24 = 0;
      if ( v22 < 0 )
      {
        v25 = v22 & 0x7FFFFFFF;
        if ( (unsigned int)v25 < *(_DWORD *)(a4 + 336) )
        {
          v26 = (_BYTE *)(*(_QWORD *)(a4 + 328) + 8 * v25);
          LOBYTE(v8) = *v26 & 1;
          v23 = (*v26 & 2) != 0;
          v24 = *(_QWORD *)v26 >> 2;
        }
      }
      return (4 * v24) | (unsigned __int8)v8 | (2LL * v23);
    }
    else
    {
      v16 = *a3;
      v17 = v15 - 6;
      if ( (*a3 & 1) != 0 )
        v18 = (((v16 >> 1) & ~(-1LL << (v16 >> 58))) >> v17) & 1;
      else
        v18 = (**(_QWORD **)v16 >> v17) & 1LL;
      if ( (_BYTE)v18 )
        return 0;
      v19 = *(_DWORD *)(v5 + 8);
      v20 = 0;
      v21 = 0;
      if ( v19 < 0 )
      {
        v27 = v19 & 0x7FFFFFFF;
        if ( (unsigned int)v27 < *(_DWORD *)(a4 + 336) )
        {
          v28 = (_BYTE *)(*(_QWORD *)(a4 + 328) + 8 * v27);
          LOBYTE(v18) = *v28 & 1;
          v20 = (*v28 & 2) != 0;
          v21 = *(_QWORD *)v28 >> 2;
        }
      }
      if ( v21 )
      {
        v29 = 1LL << v17;
        if ( (*(_BYTE *)a3 & 1) != 0 )
          *a3 = 2 * ((*a3 >> 58 << 57) | ~(-1LL << (*a3 >> 58)) & (v29 | ~(-1LL << (*a3 >> 58)) & (v16 >> 1))) + 1;
        else
          **(_QWORD **)v16 |= v29;
      }
      return (4 * v21) | (2LL * v20) | v18 & 1;
    }
  }
}
