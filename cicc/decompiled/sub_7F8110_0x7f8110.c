// Function: sub_7F8110
// Address: 0x7f8110
//
__int64 __fastcall sub_7F8110(
        char *src,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10)
{
  __int64 result; // rax
  __m128i *v15; // r12
  __int64 v16; // r12
  _QWORD *v17; // r13
  _QWORD *v18; // rax
  _QWORD *v19; // r12
  _QWORD *v20; // rax
  _QWORD *v21; // r13
  _QWORD *v22; // rax
  _QWORD *v23; // r12
  _QWORD *v24; // rax
  _QWORD *v25; // r13
  _QWORD *v26; // rax
  size_t v27; // rax
  __int64 v28; // rax
  __int64 v29; // r11
  _QWORD *v30; // r8
  __int64 *v31; // rax
  __int64 i; // rsi
  __int64 v33; // [rsp+10h] [rbp-C0h]
  _QWORD *v34; // [rsp+10h] [rbp-C0h]
  _QWORD v36[2]; // [rsp+20h] [rbp-B0h] BYREF
  __m128i v37; // [rsp+30h] [rbp-A0h]
  __m128i v38; // [rsp+40h] [rbp-90h]
  __m128i v39; // [rsp+50h] [rbp-80h]
  _BYTE v40[112]; // [rsp+60h] [rbp-70h] BYREF

  result = *(_QWORD *)a2;
  if ( !*(_QWORD *)a2 )
  {
    if ( unk_4D04358 )
    {
      v33 = sub_732700(a3, a4, a5, a6, a7, a8, a9, a10);
      v36[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
      v37 = _mm_loadu_si128(&xmmword_4F06660[1]);
      v38 = _mm_loadu_si128(&xmmword_4F06660[2]);
      v39 = _mm_loadu_si128(&xmmword_4F06660[3]);
      v36[1] = *(_QWORD *)&dword_4F077C8;
      v27 = strlen(src);
      sub_878540(src, v27);
      v37.m128i_i8[0] |= 4u;
      v28 = sub_879D20(v36, 3, v33, 0, 0, v40);
      v29 = v33;
      v30 = (_QWORD *)v28;
      if ( !v28 || *(_BYTE *)(v28 + 80) != 15 )
        goto LABEL_18;
      v31 = *(__int64 **)(v28 + 88);
      for ( i = *v31; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      while ( *(_BYTE *)(v29 + 140) == 12 )
        v29 = *(_QWORD *)(v29 + 160);
      v34 = v30;
      if ( (unsigned int)sub_8DE890(v29, i, 0, 0) && !strcmp(src, *(const char **)(*v34 + 8LL)) )
      {
        result = *(_QWORD *)(v34[11] + 8LL);
        *(_QWORD *)a2 = result;
        if ( result )
          return result;
      }
      else
      {
LABEL_18:
        *(_QWORD *)a2 = 0;
      }
    }
    v15 = sub_7F7840(src, 1, a3, 0);
    sub_7362F0((__int64)v15, 0);
    *(_QWORD *)a2 = v15;
    v15[12].m128i_i8[1] |= 0x10u;
    v16 = *(_QWORD *)(*(_QWORD *)a2 + 152LL);
    *(_BYTE *)(*(_QWORD *)(v16 + 168) + 16LL) = (2 * (dword_4F06968 == 0))
                                              | *(_BYTE *)(*(_QWORD *)(v16 + 168) + 16LL) & 0xFD;
    if ( a4 )
    {
      v17 = sub_724EF0(a4);
      **(_QWORD **)(v16 + 168) = v17;
      if ( a5 )
      {
        v18 = sub_724EF0(a5);
        *v17 = v18;
        v19 = v18;
        if ( a6 )
        {
          v20 = sub_724EF0(a6);
          *v19 = v20;
          v21 = v20;
          if ( a7 )
          {
            v22 = sub_724EF0(a7);
            *v21 = v22;
            v23 = v22;
            if ( a8 )
            {
              v24 = sub_724EF0(a8);
              *v23 = v24;
              v25 = v24;
              if ( a9 )
              {
                v26 = sub_724EF0(a9);
                *v25 = v26;
                if ( a10 )
                  *v26 = sub_724EF0(a10);
              }
            }
          }
        }
      }
    }
    return *(_QWORD *)a2;
  }
  return result;
}
