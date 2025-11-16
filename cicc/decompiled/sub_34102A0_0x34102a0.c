// Function: sub_34102A0
// Address: 0x34102a0
//
unsigned __int8 *__fastcall sub_34102A0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __int128 a8)
{
  unsigned int v8; // r15d
  unsigned int v10; // r13d
  const __m128i *v11; // r10
  unsigned __int8 *result; // rax
  const __m128i *v13; // rbx
  __int64 v14; // rax
  const __m128i *v15; // r11
  unsigned __int64 v16; // rdx
  __m128i *v17; // rax
  int v18; // esi
  _BYTE *v19; // rcx
  __int64 v20; // rdx
  __int128 v21; // [rsp-10h] [rbp-100h]
  __int64 v22; // [rsp+8h] [rbp-E8h]
  const __m128i *v23; // [rsp+10h] [rbp-E0h]
  unsigned __int64 v24; // [rsp+18h] [rbp-D8h]
  unsigned __int8 *v25; // [rsp+20h] [rbp-D0h]
  _BYTE *v26; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v27; // [rsp+38h] [rbp-B8h]
  _BYTE v28[176]; // [rsp+40h] [rbp-B0h] BYREF

  v8 = a4;
  v10 = a2;
  v11 = (const __m128i *)a8;
  if ( *((_QWORD *)&a8 + 1) == 2 )
    return sub_3406EB0(a1, a2, a3, a4, a5, a6, *(_OWORD *)a8, *(_OWORD *)(a8 + 40));
  if ( *((_QWORD *)&a8 + 1) > 2u )
  {
    if ( *((_QWORD *)&a8 + 1) == 3 )
    {
      return (unsigned __int8 *)sub_340F900(
                                  a1,
                                  a2,
                                  a3,
                                  a4,
                                  a5,
                                  a6,
                                  *(_OWORD *)a8,
                                  *(_OWORD *)(a8 + 40),
                                  *(_OWORD *)(a8 + 80));
    }
    else
    {
      v13 = (const __m128i *)a8;
      v14 = 40LL * *((_QWORD *)&a8 + 1);
      v27 = 0x800000000LL;
      v26 = v28;
      v15 = (const __m128i *)(a8 + 40LL * *((_QWORD *)&a8 + 1));
      v16 = 0xCCCCCCCCCCCCCCCDLL * ((40LL * *((_QWORD *)&a8 + 1)) >> 3);
      if ( (unsigned __int64)(40LL * *((_QWORD *)&a8 + 1)) > 0x140 )
      {
        v22 = a5;
        v23 = (const __m128i *)(a8 + v14);
        v24 = 0xCCCCCCCCCCCCCCCDLL * (v14 >> 3);
        sub_C8D5F0((__int64)&v26, v28, v16, 0x10u, a5, (__int64)v28);
        v18 = v27;
        v19 = v26;
        LODWORD(v16) = v24;
        v15 = v23;
        a5 = v22;
        v11 = (const __m128i *)a8;
        v17 = (__m128i *)&v26[16 * (unsigned int)v27];
      }
      else
      {
        v17 = (__m128i *)v28;
        v18 = 0;
        v19 = v28;
      }
      if ( v11 != v15 )
      {
        do
        {
          if ( v17 )
            *v17 = _mm_loadu_si128(v13);
          v13 = (const __m128i *)((char *)v13 + 40);
          ++v17;
        }
        while ( v15 != v13 );
        v19 = v26;
        v18 = v27;
      }
      v20 = (unsigned int)(v16 + v18);
      LODWORD(v27) = v20;
      *((_QWORD *)&v21 + 1) = v20;
      *(_QWORD *)&v21 = v19;
      result = sub_33FC220(a1, v10, a3, v8, a5, (__int64)v28, v21);
      if ( v26 != v28 )
      {
        v25 = result;
        _libc_free((unsigned __int64)v26);
        return v25;
      }
    }
  }
  else if ( *((_QWORD *)&a8 + 1) )
  {
    return sub_33FAF80((__int64)a1, a2, a3, a4, a5, a6, a7);
  }
  else
  {
    return (unsigned __int8 *)sub_33F17F0(a1, a2, a3, a4, a5);
  }
  return result;
}
