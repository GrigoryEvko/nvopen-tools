// Function: sub_374AA00
// Address: 0x374aa00
//
__int64 __fastcall sub_374AA00(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  int v5; // ecx
  __int64 v6; // rax
  int v8; // eax
  int v9; // edx
  __int64 v10; // rcx
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // r10
  unsigned int v16; // r13d
  int *v17; // r8
  int v18; // r9d
  __int64 v19; // r8
  __int64 (*v20)(); // rax
  int v21; // r8d
  int v22; // ebx
  __int64 v23; // r8

  v4 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL);
  if ( a3 != v4 )
  {
    v5 = 6;
    while ( *(_QWORD *)(v4 + 40) == *(_QWORD *)(a3 + 40) )
    {
      if ( !--v5 )
        break;
      v6 = *(_QWORD *)(v4 + 16);
      if ( !v6 || *(_QWORD *)(v6 + 8) )
        break;
      v4 = *(_QWORD *)(v6 + 24);
      if ( a3 == v4 )
        goto LABEL_9;
    }
    return 0;
  }
LABEL_9:
  if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
    return 0;
  v8 = sub_3746830(a1, a2);
  v9 = v8;
  if ( !v8 )
    return 0;
  v10 = a1[7];
  v11 = v8 < 0
      ? *(_QWORD *)(*(_QWORD *)(v10 + 56) + 16LL * (v8 & 0x7FFFFFFF) + 8)
      : *(_QWORD *)(*(_QWORD *)(v10 + 304) + 8LL * (unsigned int)v8);
  if ( !v11 )
    return 0;
  v12 = v11;
  if ( (*(_BYTE *)(v11 + 3) & 0x10) == 0 )
    goto LABEL_17;
  v12 = *(_QWORD *)(v11 + 32);
  if ( !v12 )
    return 0;
  while ( (*(_BYTE *)(v12 + 3) & 0x10) != 0 )
  {
    v12 = *(_QWORD *)(v12 + 32);
    if ( !v12 )
      return 0;
  }
LABEL_17:
  while ( 1 )
  {
    v12 = *(_QWORD *)(v12 + 32);
    if ( !v12 )
      break;
    if ( (*(_BYTE *)(v12 + 3) & 0x10) == 0 )
      return 0;
  }
  v13 = a1[5];
  v14 = *(unsigned int *)(v13 + 520);
  v15 = *(_QWORD *)(v13 + 504);
  if ( (_DWORD)v14 )
  {
    v16 = (v14 - 1) & (37 * v9);
    v17 = (int *)(v15 + 4LL * v16);
    v18 = *v17;
    if ( v9 == *v17 )
    {
LABEL_20:
      if ( v17 != (int *)(v15 + 4 * v14) )
        return 0;
    }
    else
    {
      v21 = 1;
      while ( v18 != -1 )
      {
        v22 = v21 + 1;
        v23 = ((_DWORD)v14 - 1) & (v16 + v21);
        v16 = v23;
        v17 = (int *)(v15 + 4 * v23);
        v18 = *v17;
        if ( v9 == *v17 )
          goto LABEL_20;
        v21 = v22;
      }
    }
  }
  v19 = *(_QWORD *)(v11 + 16);
  *(_QWORD *)(v13 + 752) = v19;
  *(_QWORD *)(a1[5] + 744) = *(_QWORD *)(v19 + 24);
  v20 = *(__int64 (**)())(*a1 + 16);
  if ( v20 == sub_3740E70 )
    return 0;
  return ((__int64 (__fastcall *)(__int64 *, __int64, unsigned __int64, __int64))v20)(
           a1,
           v19,
           0xCCCCCCCCCCCCCCCDLL * ((v11 - *(_QWORD *)(*(_QWORD *)(v11 + 16) + 32LL)) >> 3),
           a2);
}
