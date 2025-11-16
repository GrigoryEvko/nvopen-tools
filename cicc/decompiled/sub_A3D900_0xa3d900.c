// Function: sub_A3D900
// Address: 0xa3d900
//
__int64 __fastcall sub_A3D900(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // r9
  int v7; // ecx
  __int64 v8; // rsi
  int v9; // ecx
  unsigned int v10; // r8d
  __int64 *v11; // rdx
  __int64 v12; // r10
  unsigned int v13; // r9d
  __int64 v14; // r10
  unsigned int v15; // r8d
  __int64 *v16; // rdx
  __int64 v17; // r11
  unsigned int v18; // edx
  unsigned int v19; // r8d
  int v21; // edx
  unsigned int v22; // ecx
  unsigned int v23; // ebx
  unsigned int v24; // r8d
  unsigned int v25; // ebx
  int v26; // edx
  int v27; // ebx
  int v28; // r11d

  if ( a2 == a3 )
    return 0;
  v6 = *(_QWORD *)(a2 + 24);
  v7 = *(_DWORD *)(*(_QWORD *)a1 + 24LL);
  v8 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
  if ( !v7 )
    goto LABEL_30;
  v9 = v7 - 1;
  v10 = v9 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v11 = (__int64 *)(v8 + 16LL * v10);
  v12 = *v11;
  if ( v6 == *v11 )
  {
LABEL_4:
    v13 = *((_DWORD *)v11 + 2);
    v14 = *(_QWORD *)(a3 + 24);
  }
  else
  {
    v26 = 1;
    while ( v12 != -4096 )
    {
      v28 = v26 + 1;
      v10 = v9 & (v26 + v10);
      v11 = (__int64 *)(v8 + 16LL * v10);
      v12 = *v11;
      if ( v6 == *v11 )
        goto LABEL_4;
      v26 = v28;
    }
    v14 = *(_QWORD *)(a3 + 24);
    v13 = 0;
  }
  v15 = v9 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
  v16 = (__int64 *)(v8 + 16LL * v15);
  v17 = *v16;
  if ( *v16 == v14 )
  {
LABEL_6:
    v18 = *((_DWORD *)v16 + 2);
    if ( v18 > v13 )
    {
      v19 = 0;
      if ( v18 <= **(_DWORD **)(a1 + 8) )
        return **(unsigned __int8 **)(a1 + 16) ^ 1u;
      return v19;
    }
  }
  else
  {
    v21 = 1;
    while ( v17 != -4096 )
    {
      v27 = v21 + 1;
      v15 = v9 & (v21 + v15);
      v16 = (__int64 *)(v8 + 16LL * v15);
      v17 = *v16;
      if ( *v16 == v14 )
        goto LABEL_6;
      v21 = v27;
    }
    v18 = 0;
  }
  v22 = **(_DWORD **)(a1 + 8);
  if ( v13 > v18 )
  {
    v19 = 1;
    if ( v22 >= v13 )
      return **(unsigned __int8 **)(a1 + 16);
    return v19;
  }
  if ( v22 >= v13 )
  {
LABEL_30:
    if ( !**(_BYTE **)(a1 + 16) )
    {
      v25 = sub_BD2910(a2);
      LOBYTE(v19) = v25 < (unsigned int)sub_BD2910(a3);
      return v19;
    }
  }
  v23 = sub_BD2910(a2);
  LOBYTE(v24) = v23 > (unsigned int)sub_BD2910(a3);
  return v24;
}
