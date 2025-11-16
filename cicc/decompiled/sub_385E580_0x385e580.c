// Function: sub_385E580
// Address: 0x385e580
//
__int64 __fastcall sub_385E580(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        char a5,
        char a6,
        __m128i a7,
        __m128i a8)
{
  __int64 v9; // r13
  __int64 v10; // r9
  __int64 *v14; // r9
  char v15; // r14
  __int64 v16; // rax
  __int64 v17; // rax
  signed __int64 v18; // rsi
  __int64 v19; // rax
  unsigned int v20; // edx
  __int64 v21; // rax
  __int128 v22; // rtt
  bool v23; // al
  char v24; // al
  bool v25; // al
  char v26; // al
  bool v27; // al
  __int64 v28; // rax
  __int64 v29; // rdx
  unsigned __int8 v30; // cl
  __int64 *v31; // rdx
  __int64 *v32; // rax
  unsigned __int16 v33; // cx
  __int64 v34; // [rsp+8h] [rbp-48h]
  __int64 *v35; // [rsp+8h] [rbp-48h]
  __int64 *v36; // [rsp+8h] [rbp-48h]
  __int64 *v37; // [rsp+8h] [rbp-48h]
  bool v38; // [rsp+10h] [rbp-40h]
  __int64 v39; // [rsp+10h] [rbp-40h]
  __int64 v41; // [rsp+18h] [rbp-38h]

  v9 = *(_QWORD *)a2;
  if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)a2 + 24LL) + 8LL) - 13 <= 1 )
    return 0;
  v14 = sub_385D990(a1, a4, a2, 0, a7, a8);
  if ( *((_WORD *)v14 + 12) != 7 )
  {
    if ( !a5 )
      return 0;
    v14 = sub_14951F0(a1, a2, a7, a8);
    if ( !v14 )
      return 0;
  }
  if ( a3 != v14[6] )
    return 0;
  v38 = 0;
  if ( *(_BYTE *)(a2 + 16) == 56 )
  {
    v37 = v14;
    v25 = sub_15FA300(a2);
    v14 = v37;
    v38 = v25;
  }
  if ( !a6 )
    goto LABEL_9;
  v35 = v14;
  v23 = sub_1494FF0(a1, a2, 1, a7, a8);
  v14 = v35;
  v15 = v23;
  if ( v23 || (*((_BYTE *)v35 + 26) & 7) != 0 )
    goto LABEL_9;
  if ( *(_BYTE *)(a2 + 16) == 56 )
  {
    v27 = sub_15FA300(a2);
    v14 = v35;
    if ( v27 )
    {
      v28 = a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      if ( a2 != v28 )
      {
        v29 = 0;
        do
        {
          if ( *(_BYTE *)(*(_QWORD *)v28 + 16LL) != 13 )
          {
            if ( v29 )
              goto LABEL_25;
            v29 = *(_QWORD *)v28;
          }
          v28 += 24;
        }
        while ( a2 != v28 );
        if ( v29 )
        {
          v30 = *(_BYTE *)(v29 + 16);
          if ( v30 <= 0x17u )
          {
            if ( v30 != 5 )
              goto LABEL_25;
            v33 = *(_WORD *)(v29 + 18);
            if ( v33 > 0x17u || (((unsigned __int64)&loc_80A800 >> v33) & 1) == 0 )
              goto LABEL_25;
          }
          else if ( v30 > 0x2Fu || ((0x80A800000000uLL >> v30) & 1) == 0 )
          {
            goto LABEL_25;
          }
          if ( (*(_BYTE *)(v29 + 17) & 4) != 0 )
          {
            v31 = (*(_BYTE *)(v29 + 23) & 0x40) != 0
                ? *(__int64 **)(v29 - 8)
                : (__int64 *)(v29 - 24LL * (*(_DWORD *)(v29 + 20) & 0xFFFFFFF));
            if ( *(_BYTE *)(v31[3] + 16) == 13 )
            {
              v32 = sub_1494E70(a1, *v31, a7, a8);
              v14 = v35;
              if ( *((_WORD *)v32 + 12) == 7 && a3 == v32[6] && (*((_BYTE *)v32 + 26) & 4) != 0 )
              {
LABEL_9:
                v15 = 1;
                goto LABEL_10;
              }
            }
          }
        }
      }
    }
  }
LABEL_25:
  if ( !v38 )
  {
    v36 = v14;
    v24 = sub_15E4690(*(_QWORD *)(**(_QWORD **)(a3 + 32) + 56LL), *(_DWORD *)(v9 + 8) >> 8);
    v14 = v36;
    if ( v24 )
    {
      v15 = a5;
      if ( !a5 )
        return 0;
      sub_1497CC0(a1, a2, 1, a7, a8);
      v14 = v36;
    }
  }
LABEL_10:
  v34 = sub_13A5BC0(v14, *(_QWORD *)(a1 + 112));
  if ( *(_WORD *)(v34 + 24) )
    return 0;
  v16 = sub_157EB90(**(_QWORD **)(a3 + 32));
  v17 = sub_1632FA0(v16);
  v18 = sub_12BE0A0(v17, *(_QWORD *)(v9 + 24));
  v19 = *(_QWORD *)(v34 + 32);
  v20 = *(_DWORD *)(v19 + 32);
  if ( v20 > 0x40 )
    return 0;
  v22 = (__int64)(*(_QWORD *)(v19 + 24) << (64 - (unsigned __int8)v20)) >> (64 - (unsigned __int8)v20);
  v21 = ((__int64)(*(_QWORD *)(v19 + 24) << (64 - (unsigned __int8)v20)) >> (64 - (unsigned __int8)v20)) / v18;
  if ( (unsigned __int64)(v22 % v18) )
    return 0;
  v10 = v21;
  if ( v15 != 1 && v21 != 1 && v21 != -1 )
  {
    if ( v38
      || (v39 = v21,
          v26 = sub_15E4690(*(_QWORD *)(**(_QWORD **)(a3 + 32) + 56LL), *(_DWORD *)(v9 + 8) >> 8),
          v10 = v39,
          !v26) )
    {
      if ( a5 )
      {
        v41 = v10;
        sub_1497CC0(a1, a2, 1, a7, a8);
        return v41;
      }
      return 0;
    }
  }
  return v10;
}
