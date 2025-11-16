// Function: sub_1BB46B0
// Address: 0x1bb46b0
//
__int64 __fastcall sub_1BB46B0(__int64 a1, char a2, unsigned int a3, unsigned int a4, __m128i a5, __m128i a6)
{
  unsigned int v7; // r12d
  int v11; // r12d
  int v12; // eax
  unsigned int v13; // r15d
  unsigned int v14; // esi
  unsigned int v15; // r12d
  unsigned __int64 v16; // r15
  unsigned __int64 v17; // rcx
  unsigned int v18; // esi
  unsigned __int64 v19; // rcx
  __int64 v20; // rsi
  int v21; // edi
  unsigned __int64 v22; // r8
  unsigned __int64 v23; // rcx
  __int64 v24; // rsi
  unsigned int v25; // ecx
  unsigned int v26; // r9d
  unsigned int v27; // r10d
  unsigned int v28; // r9d
  unsigned int v29; // eax
  unsigned int v30; // esi
  _QWORD *v31; // rdx
  unsigned int v32; // ecx
  int v33; // [rsp+Ch] [rbp-94h]
  unsigned int v34; // [rsp+1Ch] [rbp-84h] BYREF
  int *v35; // [rsp+20h] [rbp-80h] BYREF
  char v36; // [rsp+30h] [rbp-70h] BYREF

  if ( a2
    || *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 320) + 48LL) + 40LL) != -1
    || (unsigned int)sub_1474220(*(_QWORD *)(*(_QWORD *)(a1 + 304) + 112LL), *(_QWORD *)(a1 + 296)) - 2 <= 0x7D )
  {
    return 1;
  }
  v11 = sub_14A3140(*(__int64 **)(a1 + 328), a3 > 1);
  if ( a3 == 1 )
  {
    if ( (int)sub_1B907A0(dword_4FB8A28) > 0 )
      v11 = dword_4FB8AC0;
  }
  else if ( (int)sub_1B907A0(dword_4FB8948) > 0 )
  {
    v11 = dword_4FB89E0;
  }
  v34 = a3;
  sub_1BB2200(&v35, (_QWORD *)a1, (__int64)&v34, 1u, a5, a6);
  v12 = *v35;
  v13 = v35[1];
  if ( v35 != (int *)&v36 )
  {
    v33 = *v35;
    _libc_free((unsigned __int64)v35);
    v12 = v33;
  }
  v14 = 1;
  if ( v13 )
    v14 = v13;
  v15 = v11 - v12;
  LODWORD(v16) = 0;
  if ( v15 / v14 )
  {
    _BitScanReverse64(&v17, v15 / v14);
    v16 = 0x8000000000000000LL >> ((unsigned __int8)v17 ^ 0x3Fu);
  }
  if ( byte_4FB82E0 )
  {
    v18 = v14 - 1;
    if ( !v18 )
      v18 = 1;
    LODWORD(v16) = 0;
    if ( (v15 - 1) / v18 )
    {
      _BitScanReverse64(&v19, (v15 - 1) / v18);
      v16 = 0x8000000000000000LL >> ((unsigned __int8)v19 ^ 0x3Fu);
    }
  }
  v7 = sub_14A3320(*(_QWORD *)(a1 + 328));
  if ( a3 == 1 )
  {
    if ( (int)sub_1B907A0(dword_4FB8868) > 0 )
      v7 = dword_4FB8900;
  }
  else if ( (int)sub_1B907A0(dword_4FB8788) > 0 )
  {
    v7 = dword_4FB8820;
  }
  if ( !a4 )
    a4 = sub_1BA8260(a1, a3);
  if ( v7 >= (unsigned int)v16 )
  {
    v7 = 1;
    if ( (_DWORD)v16 )
      v7 = v16;
  }
  v20 = *(_QWORD *)(a1 + 320);
  v21 = *(_DWORD *)(v20 + 88);
  if ( a3 <= 1 )
  {
    if ( a3 == 1 && **(_BYTE **)(*(_QWORD *)(v20 + 48) + 8LL) )
      goto LABEL_58;
  }
  else if ( v21 )
  {
    return v7;
  }
  if ( dword_4FB8660 > a4 )
  {
    LODWORD(v22) = 0;
    if ( dword_4FB8660 / a4 )
    {
      _BitScanReverse64(&v23, dword_4FB8660 / a4);
      v22 = 0x8000000000000000LL >> ((unsigned __int8)v23 ^ 0x3Fu);
      if ( v7 < (unsigned int)v22 )
        LODWORD(v22) = v7;
    }
    v24 = *(_QWORD *)(v20 + 48);
    v25 = 1;
    v26 = 1;
    if ( *(_DWORD *)(v24 + 36) )
      v26 = *(_DWORD *)(v24 + 36);
    if ( *(_DWORD *)(v24 + 32) )
      v25 = *(_DWORD *)(v24 + 32);
    v27 = v7 / v26;
    v28 = v7 / v26;
    v29 = v7 / v25;
    v30 = v7 / v25;
    if ( v21 )
    {
      v31 = **(_QWORD ***)(a1 + 296);
      if ( v31 )
      {
        v32 = 1;
        do
        {
          v31 = (_QWORD *)*v31;
          ++v32;
        }
        while ( v31 );
        if ( v32 > 1 )
        {
          v30 = dword_4FB8120;
          if ( (unsigned int)v22 > dword_4FB8120 )
            LODWORD(v22) = dword_4FB8120;
          if ( v27 > dword_4FB8120 )
            v27 = dword_4FB8120;
          if ( v29 <= dword_4FB8120 )
            v30 = v29;
          v28 = v27;
        }
      }
    }
    v7 = v22;
    if ( byte_4FB84A0 )
    {
      if ( v30 < v28 )
        v30 = v28;
      if ( v30 >= (unsigned int)v22 )
        return v30;
    }
    return v7;
  }
LABEL_58:
  if ( !(unsigned __int8)sub_14A2ED0(*(__int64 **)(a1 + 328), v21 != 0) )
    return 1;
  return v7;
}
