// Function: sub_1B1D750
// Address: 0x1b1d750
//
__int64 __fastcall sub_1B1D750(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r12d
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rdi
  unsigned int v13; // ecx
  int v14; // eax
  __int64 v15; // r10
  __int64 i; // rdx
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // r14
  int v23; // eax
  int v24; // edx
  __int64 v25; // rsi
  unsigned int v26; // eax
  __int64 v27; // rcx
  int v28; // edi
  __int64 v29; // r12
  _QWORD *v30; // rax
  int v31; // edx
  _QWORD *v32; // rdx
  __int64 v33; // rdi
  _QWORD *v34; // rax
  unsigned int v35; // eax
  __int64 v36[7]; // [rsp+8h] [rbp-38h] BYREF

  if ( *(_QWORD *)(a1 + 40) != **(_QWORD **)(a2 + 32) )
    return 0;
  if ( (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) != 2 )
    return 0;
  v10 = sub_13FC520(a2);
  v11 = sub_13FCB50(a2);
  v12 = v11;
  if ( !v10 )
    return 0;
  if ( !v11 )
    return 0;
  v13 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( !v13 )
    return 0;
  v14 = 0;
  v15 = 24LL * *(unsigned int *)(a1 + 56);
  for ( i = v15 + 8; ; i += 8 )
  {
    v17 = a1 - 24LL * v13;
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v17 = *(_QWORD *)(a1 - 8);
    if ( v10 == *(_QWORD *)(v17 + i) )
      break;
    if ( v13 == ++v14 )
      return 0;
  }
  if ( v14 < 0 )
    return 0;
  v18 = 0;
  v19 = v17 + v15;
  while ( v12 != *(_QWORD *)(v19 + 8 * v18 + 8) )
  {
    if ( v13 == (_DWORD)++v18 )
      return 0;
  }
  if ( (int)v18 < 0 )
    return 0;
  v20 = 0;
  do
  {
    if ( v12 == *(_QWORD *)(v19 + 8 * v20 + 8) )
    {
      v21 = 24 * v20;
      goto LABEL_22;
    }
    ++v20;
  }
  while ( v13 != (_DWORD)v20 );
  v21 = 0x17FFFFFFE8LL;
LABEL_22:
  v22 = *(_QWORD *)(v17 + v21);
  if ( *(_BYTE *)(v22 + 16) <= 0x17u || !sub_1377F70(a2 + 56, *(_QWORD *)(v22 + 40)) || *(_BYTE *)(v22 + 16) == 77 )
    return 0;
  v23 = *(_DWORD *)(a3 + 24);
  if ( v23 )
  {
    v24 = v23 - 1;
    v25 = *(_QWORD *)(a3 + 8);
    v26 = (v23 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
    v27 = *(_QWORD *)(v25 + 16LL * v26);
    if ( v27 == v22 )
      return 0;
    v28 = 1;
    while ( v27 != -8 )
    {
      v26 = v24 & (v28 + v26);
      v27 = *(_QWORD *)(v25 + 16LL * v26);
      if ( v22 == v27 )
        return 0;
      ++v28;
    }
  }
  v29 = *(_QWORD *)(a1 + 8);
  if ( !v29 )
    return 1;
  if ( *(_QWORD *)(v29 + 8)
    || (v30 = sub_1648700(*(_QWORD *)(a1 + 8)),
        v31 = *((unsigned __int8 *)v30 + 16),
        v36[0] = (__int64)v30,
        (unsigned int)(v31 - 60) > 0xC)
    || *(_QWORD *)(a1 + 40) != v30[5]
    || (v33 = v30[1]) == 0
    || *(_QWORD *)(v33 + 8) )
  {
LABEL_33:
    while ( 1 )
    {
      v32 = sub_1648700(v29);
      if ( *((_BYTE *)v32 + 16) > 0x17u && !sub_15CCEE0(a4, v22, (__int64)v32) )
        break;
      v29 = *(_QWORD *)(v29 + 8);
      if ( !v29 )
        return 1;
    }
    return 0;
  }
  v34 = sub_1648700(v33);
  LOBYTE(v35) = sub_15CCEE0(a4, v22, (__int64)v34);
  v4 = v35;
  if ( !(_BYTE)v35 )
  {
    v29 = *(_QWORD *)(a1 + 8);
    if ( v29 )
      goto LABEL_33;
    return 1;
  }
  if ( !sub_15CCEE0(a4, v22, v36[0]) )
    sub_1B1D510(a3, v36)[1] = v22;
  return v4;
}
