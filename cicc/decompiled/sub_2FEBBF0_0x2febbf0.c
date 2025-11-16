// Function: sub_2FEBBF0
// Address: 0x2febbf0
//
__int64 __fastcall sub_2FEBBF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  int v7; // r13d
  __int16 v10; // r14
  int v11; // ecx
  __int64 *v12; // rdi
  __int64 v13; // rax
  int v15; // eax
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rsi
  unsigned int v19; // edx
  unsigned __int16 v20; // dx
  unsigned __int64 v21; // rdi
  unsigned __int16 v22; // dx
  __int64 v23; // [rsp+0h] [rbp-60h] BYREF
  __int64 v24; // [rsp+8h] [rbp-58h]
  _QWORD v25[3]; // [rsp+10h] [rbp-50h] BYREF
  int v26[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v7 = a2;
  v25[0] = a2;
  v25[1] = a3;
  v23 = a4;
  v24 = a5;
  if ( (_WORD)a2 )
  {
    if ( (unsigned __int16)(a2 - 17) > 0x9Eu )
    {
      v10 = v23;
LABEL_4:
      if ( v10 )
        goto LABEL_5;
      return 1;
    }
  }
  else if ( !(unsigned __int8)sub_30070D0(v25) )
  {
    return 1;
  }
  v10 = v23;
  if ( !(_WORD)v23 )
  {
    if ( !(unsigned __int8)sub_30070D0(&v23) )
      return 1;
    if ( (unsigned __int8)sub_3007100(&v23) )
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
    v15 = sub_3007130(&v23);
LABEL_23:
    if ( v15 == 1 )
      return 0;
    if ( !(_WORD)a2 )
      return 1;
    goto LABEL_4;
  }
  if ( (unsigned __int16)(v23 - 17) <= 0x9Eu )
  {
    v15 = word_4456340[(unsigned __int16)v23 - 1];
    goto LABEL_23;
  }
  if ( !(_WORD)a2 )
    return 1;
LABEL_5:
  v11 = (unsigned __int16)a2;
  if ( *(_BYTE *)(a1 + 500LL * (unsigned __int16)a2 + 6712) != 1 )
    goto LABEL_7;
  v16 = *(_QWORD *)(a1 + 525256);
  v17 = a1 + 525248;
  if ( !v16 )
    goto LABEL_35;
  v18 = a1 + 525248;
  do
  {
    while ( 1 )
    {
      v19 = *(_DWORD *)(v16 + 32);
      if ( v19 <= 0x129 || v19 == 298 && (unsigned __int16)v7 > *(_WORD *)(v16 + 36) )
        break;
      v18 = v16;
      v16 = *(_QWORD *)(v16 + 16);
      if ( !v16 )
        goto LABEL_33;
    }
    v16 = *(_QWORD *)(v16 + 24);
  }
  while ( v16 );
LABEL_33:
  if ( v17 == v18
    || *(_DWORD *)(v18 + 32) > 0x12Au
    || *(_DWORD *)(v18 + 32) == 298 && (unsigned __int16)v7 < *(_WORD *)(v18 + 36) )
  {
LABEL_35:
    v20 = v7;
    if ( (unsigned __int16)(v7 - 17) <= 0xD3u )
      v20 = word_4456580[v11 - 1];
    if ( v20 <= 1u || (unsigned __int16)(v20 - 504) <= 7u )
LABEL_54:
      BUG();
    v21 = *(_QWORD *)&byte_444C4A0[16 * v20 - 16];
    do
    {
      ++v7;
      while ( 1 )
      {
        v22 = v7;
        if ( (unsigned __int16)(v7 - 17) <= 0xD3u )
          v22 = word_4456580[(unsigned __int16)v7 - 1];
        if ( v22 <= 1u || (unsigned __int16)(v22 - 504) <= 7u )
          goto LABEL_54;
        if ( v21 < *(_QWORD *)&byte_444C4A0[16 * v22 - 16] )
          break;
        ++v7;
      }
    }
    while ( !(_WORD)v7
         || !*(_QWORD *)(a1 + 8LL * (unsigned __int16)v7 + 112)
         || *(_BYTE *)(a1 + 500LL * (unsigned __int16)v7 + 6712) == 1 );
  }
  else
  {
    LOWORD(v7) = *(_WORD *)(v18 + 40);
  }
  if ( (_WORD)v7 != v10 )
  {
LABEL_7:
    v12 = *(__int64 **)(a6 + 40);
    v26[0] = 0;
    v13 = sub_2E79000(v12);
    if ( (unsigned __int8)sub_2FEBB30(a1, *(_QWORD *)(a6 + 64), v13, (unsigned int)v23, v24, a7, v26) )
    {
      if ( v26[0] )
        return 1;
    }
  }
  return 0;
}
