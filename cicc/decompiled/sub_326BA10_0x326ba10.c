// Function: sub_326BA10
// Address: 0x326ba10
//
__int64 __fastcall sub_326BA10(
        __int64 a1,
        __int64 a2,
        int a3,
        int a4,
        char a5,
        __int64 a6,
        __int64 a7,
        int a8,
        char a9,
        int a10)
{
  __int64 v11; // rax
  int v13; // ecx
  __int64 v14; // rax
  __int64 result; // rax
  __int64 v16; // rax
  __int64 (*v17)(); // rax
  int v18; // r9d
  __int64 v19; // rsi
  __int128 v20; // rax
  __int64 v21; // r10
  __int64 v22; // r14
  __int64 v23; // [rsp+20h] [rbp-40h] BYREF
  int v24; // [rsp+28h] [rbp-38h]

  v11 = *(_QWORD *)(a7 + 56);
  if ( !v11 )
    return 0;
  v13 = 1;
  do
  {
    while ( a8 != *(_DWORD *)(v11 + 8) )
    {
      v11 = *(_QWORD *)(v11 + 32);
      if ( !v11 )
        goto LABEL_9;
    }
    if ( !v13 )
      return 0;
    v14 = *(_QWORD *)(v11 + 32);
    if ( !v14 )
      goto LABEL_10;
    if ( a8 == *(_DWORD *)(v14 + 8) )
      return 0;
    v11 = *(_QWORD *)(v14 + 32);
    v13 = 0;
  }
  while ( v11 );
LABEL_9:
  if ( v13 == 1 )
    return 0;
LABEL_10:
  if ( *(_DWORD *)(a7 + 24) != 362 || (*(_BYTE *)(a7 + 33) & 0xC) != 0 )
    return 0;
  if ( a5 || (*(_BYTE *)(*(_QWORD *)(a7 + 112) + 37LL) & 0xF) != 0 || (*(_BYTE *)(a7 + 32) & 8) != 0 )
  {
    if ( !(_WORD)a3 )
      return 0;
    v16 = **(unsigned __int16 **)(a7 + 48);
    if ( !(_WORD)v16
      || (((int)*(unsigned __int16 *)(a2 + 2 * (v16 + 274LL * (unsigned __int16)a3 + 71704) + 6) >> (4 * a9)) & 0xB) != 0 )
    {
      return 0;
    }
  }
  v17 = *(__int64 (**)())(*(_QWORD *)a2 + 1584LL);
  if ( v17 == sub_2FE3520 || !((unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD))v17)(a2, a6, 0) )
    return 0;
  v19 = *(_QWORD *)(a7 + 80);
  v23 = v19;
  if ( v19 )
    sub_B96E90((__int64)&v23, v19, 1);
  v24 = *(_DWORD *)(a7 + 72);
  *(_QWORD *)&v20 = sub_33FAF80(a1, a10, (unsigned int)&v23, a3, a4, v18, *(_OWORD *)(*(_QWORD *)(a7 + 40) + 160LL));
  v21 = *(_QWORD *)(a7 + 40);
  v22 = sub_33E8F60(
          a1,
          a3,
          a4,
          (unsigned int)&v23,
          *(_QWORD *)v21,
          *(_QWORD *)(v21 + 8),
          *(_QWORD *)(v21 + 40),
          *(_QWORD *)(v21 + 48),
          *(_OWORD *)(v21 + 80),
          *(_OWORD *)(v21 + 120),
          v20,
          *(unsigned __int16 *)(a7 + 96),
          *(_QWORD *)(a7 + 104),
          *(_QWORD *)(a7 + 112),
          (*(_WORD *)(a7 + 32) >> 7) & 7,
          a9,
          (*(_BYTE *)(a7 + 33) & 0x10) != 0);
  sub_34161C0(a1, a7, 1, v22, 1);
  result = v22;
  if ( v23 )
  {
    sub_B91220((__int64)&v23, v23);
    return v22;
  }
  return result;
}
