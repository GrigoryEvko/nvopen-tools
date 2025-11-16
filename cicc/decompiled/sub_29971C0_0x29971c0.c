// Function: sub_29971C0
// Address: 0x29971c0
//
__int64 __fastcall sub_29971C0(__int64 a1, unsigned __int8 *a2, _QWORD *a3)
{
  unsigned int v3; // r15d
  unsigned __int8 **v5; // rbx
  __int64 v6; // rdx
  unsigned __int8 **v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rdx
  unsigned __int8 **v10; // rdx
  __int64 v11; // rdx
  __m128i v13; // xmm0
  __m128i v14; // xmm1
  __m128i v15; // xmm2
  __int64 *v16; // rax
  __int64 *v17; // rax
  char v18; // r12
  unsigned __int64 v19; // rax
  __int64 v20; // [rsp+8h] [rbp-358h]
  __m128i v21; // [rsp+20h] [rbp-340h] BYREF
  __m128i v22; // [rsp+30h] [rbp-330h] BYREF
  __m128i v23; // [rsp+40h] [rbp-320h] BYREF
  __m128i v24[3]; // [rsp+50h] [rbp-310h] BYREF
  char v25; // [rsp+80h] [rbp-2E0h]
  _QWORD v26[2]; // [rsp+90h] [rbp-2D0h] BYREF
  __int64 v27; // [rsp+A0h] [rbp-2C0h]
  __int64 v28; // [rsp+A8h] [rbp-2B8h] BYREF
  unsigned int v29; // [rsp+B0h] [rbp-2B0h]
  _QWORD v30[2]; // [rsp+1E8h] [rbp-178h] BYREF
  char v31; // [rsp+1F8h] [rbp-168h]
  _BYTE *v32; // [rsp+200h] [rbp-160h]
  __int64 v33; // [rsp+208h] [rbp-158h]
  _BYTE v34[128]; // [rsp+210h] [rbp-150h] BYREF
  __int16 v35; // [rsp+290h] [rbp-D0h]
  _QWORD v36[2]; // [rsp+298h] [rbp-C8h] BYREF
  __int64 v37; // [rsp+2A8h] [rbp-B8h]
  __int64 v38; // [rsp+2B0h] [rbp-B0h] BYREF
  unsigned int v39; // [rsp+2B8h] [rbp-A8h]
  char v40; // [rsp+330h] [rbp-30h] BYREF

  if ( *(_BYTE *)a1 == 85 )
  {
    v11 = *(_QWORD *)(a1 - 32);
    if ( !v11 )
      goto LABEL_20;
    if ( !*(_BYTE *)v11
      && *(_QWORD *)(v11 + 24) == *(_QWORD *)(a1 + 80)
      && (*(_BYTE *)(v11 + 33) & 0x20) != 0
      && (unsigned int)(*(_DWORD *)(v11 + 36) - 68) <= 3 )
    {
      return 1;
    }
    if ( *(_BYTE *)v11
      || *(_QWORD *)(v11 + 24) != *(_QWORD *)(a1 + 80)
      || (*(_BYTE *)(v11 + 33) & 0x20) == 0
      || *(_DWORD *)(v11 + 36) != 210 )
    {
LABEL_20:
      v5 = (unsigned __int8 **)a1;
      if ( !(unsigned __int8)sub_B46970((unsigned __int8 *)a1) )
        goto LABEL_5;
      return 0;
    }
    if ( sub_98C100(*(_QWORD *)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF))), 0) )
      return 1;
  }
  v3 = sub_B46970((unsigned __int8 *)a1);
  if ( (_BYTE)v3 )
    return 0;
  v5 = (unsigned __int8 **)a1;
  if ( *(_BYTE *)a1 != 61 || !(unsigned __int8)sub_B46970(a2) )
    goto LABEL_5;
  v20 = sub_B43CC0(a1);
  sub_D665A0(&v21, a1);
  v13 = _mm_loadu_si128(&v21);
  v14 = _mm_loadu_si128(&v22);
  v25 = 1;
  v15 = _mm_loadu_si128(&v23);
  v26[0] = a3;
  v16 = &v28;
  v26[1] = 0;
  v27 = 1;
  v24[0] = v13;
  v24[1] = v14;
  v24[2] = v15;
  do
  {
    *v16 = -4;
    v16 += 5;
    *(v16 - 4) = -3;
    *(v16 - 3) = -4;
    *(v16 - 2) = -3;
  }
  while ( v16 != v30 );
  v30[1] = 0;
  v32 = v34;
  v33 = 0x400000000LL;
  v30[0] = v36;
  v31 = 0;
  v35 = 256;
  v36[1] = 0;
  v37 = 1;
  v36[0] = &unk_49DDBE8;
  v17 = &v38;
  do
  {
    *v17 = -4096;
    v17 += 2;
  }
  while ( v17 != (__int64 *)&v40 );
  v18 = sub_CF63E0(a3, a2, v24, (__int64)v26);
  v36[0] = &unk_49DDBE8;
  if ( (v37 & 1) == 0 )
    sub_C7D6A0(v38, 16LL * v39, 8);
  nullsub_184();
  if ( v32 != v34 )
    _libc_free((unsigned __int64)v32);
  if ( (v27 & 1) == 0 )
    sub_C7D6A0(v28, 40LL * v29, 8);
  if ( (v18 & 2) == 0 )
  {
    _BitScanReverse64(&v19, 1LL << (*(_WORD *)(a1 + 2) >> 1));
    if ( (unsigned __int8)sub_D31180(
                            *(_QWORD *)(a1 - 32),
                            *(_QWORD *)(a1 + 8),
                            63 - ((unsigned int)v19 ^ 0x3F),
                            v20,
                            a1,
                            0,
                            0,
                            0) )
    {
LABEL_5:
      v6 = 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
      v7 = (unsigned __int8 **)(a1 - v6);
      if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
      {
        v7 = *(unsigned __int8 ***)(a1 - 8);
        v5 = &v7[(unsigned __int64)v6 / 8];
      }
      v8 = v6 >> 5;
      v9 = v6 >> 7;
      if ( v9 )
      {
        v10 = &v7[16 * v9];
        while ( a2 != *v7 )
        {
          if ( a2 == v7[4] )
          {
            LOBYTE(v3) = v5 == v7 + 4;
            return v3;
          }
          if ( a2 == v7[8] )
          {
            LOBYTE(v3) = v5 == v7 + 8;
            return v3;
          }
          if ( a2 == v7[12] )
          {
            LOBYTE(v3) = v5 == v7 + 12;
            return v3;
          }
          v7 += 16;
          if ( v10 == v7 )
          {
            v8 = ((char *)v5 - (char *)v7) >> 5;
            goto LABEL_37;
          }
        }
        goto LABEL_14;
      }
LABEL_37:
      if ( v8 != 2 )
      {
        if ( v8 != 3 )
        {
          if ( v8 != 1 )
            return 1;
LABEL_48:
          v3 = 1;
          if ( a2 != *v7 )
            return v3;
          goto LABEL_14;
        }
        if ( a2 == *v7 )
        {
LABEL_14:
          LOBYTE(v3) = v5 == v7;
          return v3;
        }
        v7 += 4;
      }
      if ( a2 != *v7 )
      {
        v7 += 4;
        goto LABEL_48;
      }
      goto LABEL_14;
    }
  }
  return v3;
}
