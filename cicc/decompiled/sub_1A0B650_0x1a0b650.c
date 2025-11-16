// Function: sub_1A0B650
// Address: 0x1a0b650
//
__int64 __fastcall sub_1A0B650(
        __int64 a1,
        __int64 a2,
        _DWORD *a3,
        __m128 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11,
        __int64 a12,
        int a13)
{
  unsigned int v15; // eax
  int v16; // r9d
  __int64 v17; // r12
  unsigned int v18; // r15d
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rsi
  int v23; // r13d
  __int64 v24; // rax
  __int64 v25; // rax
  _QWORD *v26; // rax
  __int64 v27; // rax
  int v28; // [rsp+Ch] [rbp-34h]
  int v29; // [rsp+Ch] [rbp-34h]

  v15 = a3[2];
  while ( 1 )
  {
    v16 = *(unsigned __int8 *)(a2 + 16);
    v17 = 0;
    v18 = v16 - 24;
LABEL_3:
    v19 = 16LL * v15 - 16;
    while ( 1 )
    {
      if ( !v15 )
        return v17;
      v20 = *(_QWORD *)a3;
      v21 = *(_QWORD *)(*(_QWORD *)a3 + v19 + 8);
      if ( *(_BYTE *)(v21 + 16) > 0x10u )
        break;
      --v15;
      v19 -= 16;
      a3[2] = v15;
      if ( v17 )
      {
        v29 = v16;
        v27 = sub_15A2A30((__int64 *)v18, (__int64 *)v21, v17, 0, 0, *(double *)a4.m128_u64, *(double *)a5.m128_u64, a6);
        v16 = v29;
        v17 = v27;
        v15 = a3[2];
        goto LABEL_3;
      }
      v17 = v21;
    }
    v23 = v15;
    if ( v17 )
      break;
LABEL_15:
    if ( v23 == 1 )
      return *(_QWORD *)(*(_QWORD *)a3 + 8LL);
LABEL_16:
    if ( v18 > 0x1B )
    {
      if ( v18 != 28 )
        return 0;
      v17 = sub_1A08F70(a1, a2, (__int64)a3, v20);
      if ( v17 )
        return v17;
    }
    else if ( v18 > 0x19 )
    {
      v17 = sub_19FF700(v18, (__int64)a3, v19, v20);
      if ( v17 )
        return v17;
    }
    else if ( v18 > 0xC )
    {
      if ( (unsigned int)(v16 - 39) > 1 )
        return 0;
      v17 = sub_1A07880(a1, a2, (__int64)a3, a4, a5, a6, v20, a13);
      if ( v17 )
        return v17;
    }
    else
    {
      if ( v18 <= 0xA )
        return 0;
      v17 = sub_1A0A170(a1, a2, (__int64)a3, a4, *(double *)a5.m128_u64, a6, a7, a8, a9, a10, a11);
      if ( v17 )
        return v17;
    }
    v15 = a3[2];
    if ( v15 == v23 )
      return 0;
  }
  v28 = v16;
  v24 = sub_15A14F0(v18, *(__int64 ***)a2, 0);
  v16 = v28;
  if ( v17 != v24 )
  {
    if ( v17 == sub_15A0900(v18, *(__int64 ***)a2, v19, v20) )
      return v17;
    v25 = (unsigned int)a3[2];
    v16 = v28;
    if ( (unsigned int)v25 >= a3[3] )
    {
      sub_16CD150((__int64)a3, a3 + 4, 0, 16, a13, v28);
      v25 = (unsigned int)a3[2];
      v16 = v28;
    }
    v26 = (_QWORD *)(*(_QWORD *)a3 + 16 * v25);
    *v26 = 0;
    v26[1] = v17;
    v23 = a3[2] + 1;
    a3[2] = v23;
    goto LABEL_15;
  }
  v23 = a3[2];
  if ( v23 != 1 )
    goto LABEL_16;
  return *(_QWORD *)(*(_QWORD *)a3 + 8LL);
}
