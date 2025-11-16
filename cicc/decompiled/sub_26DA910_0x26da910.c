// Function: sub_26DA910
// Address: 0x26da910
//
__int64 __fastcall sub_26DA910(__int64 a1, __int64 a2, __m128i *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // rcx
  __int64 v8; // r8
  __m128i *v9; // rbx
  __int64 v10; // rdx
  __m128i *v11; // rax
  __m128i *v13; // rax
  char v14; // dl
  __m128i *v15; // r14
  _QWORD *i; // rax
  _QWORD *v17; // rdx
  __m128i *v18; // rax
  _QWORD *v19; // rcx
  bool v20; // r10
  unsigned __int64 v21; // rax
  __m128i si128; // xmm0
  __int64 v23; // rdx
  __int64 v24; // rax
  __m128i *v25; // rax
  _QWORD *v26; // [rsp+8h] [rbp-58h]
  char v27; // [rsp+14h] [rbp-4Ch]
  _QWORD *v28; // [rsp+18h] [rbp-48h]
  __m128i v29[4]; // [rsp+20h] [rbp-40h] BYREF

  if ( *(_QWORD *)(a2 + 568) )
  {
    v13 = sub_26D8D40(a2 + 528, a3);
    *(_BYTE *)(a1 + 8) = 0;
    *(_QWORD *)a1 = v13;
    *(_BYTE *)(a1 + 16) = v14;
  }
  else
  {
    v7 = *(unsigned int *)(a2 + 8);
    v8 = *(_QWORD *)a2;
    v9 = (__m128i *)(*(_QWORD *)a2 + 16 * v7);
    if ( *(__m128i **)a2 == v9 )
    {
      if ( v7 > 0x1F )
      {
        v28 = (_QWORD *)(a2 + 528);
LABEL_27:
        *(_DWORD *)(a2 + 8) = 0;
        v25 = sub_26D8D40((__int64)v28, a3);
        *(_BYTE *)(a1 + 8) = 0;
        *(_QWORD *)a1 = v25;
        *(_BYTE *)(a1 + 16) = 1;
        return a1;
      }
    }
    else
    {
      v10 = a3->m128i_i64[0];
      v11 = *(__m128i **)a2;
      while ( v11->m128i_i64[0] != v10 || v11->m128i_i64[1] != a3->m128i_i64[1] )
      {
        if ( v9 == ++v11 )
          goto LABEL_11;
      }
      if ( v9 != v11 )
      {
        *(_BYTE *)(a1 + 8) = 1;
        *(_QWORD *)a1 = v11;
        *(_BYTE *)(a1 + 16) = 0;
        return a1;
      }
LABEL_11:
      if ( v7 > 0x1F )
      {
        v15 = *(__m128i **)a2;
        v28 = (_QWORD *)(a2 + 528);
        v29[0].m128i_i64[0] = a2 + 536;
        for ( i = sub_26DA7C0((_QWORD *)(a2 + 528), (_QWORD *)(a2 + 536), (unsigned __int64 *)v15);
              ;
              i = sub_26DA7C0(v28, v29[0].m128i_i64[0], (unsigned __int64 *)v15) )
        {
          if ( v17 )
          {
            v20 = 1;
            if ( !i && v17 != (_QWORD *)v29[0].m128i_i64[0] )
            {
              v21 = v17[4];
              if ( v15->m128i_i64[0] >= v21 )
              {
                v20 = 0;
                if ( v15->m128i_i64[0] == v21 )
                  v20 = v15->m128i_i64[1] < v17[5];
              }
            }
            v26 = v17;
            v27 = v20;
            v18 = (__m128i *)sub_22077B0(0x30u);
            v19 = (_QWORD *)v29[0].m128i_i64[0];
            v18[2] = _mm_loadu_si128(v15);
            sub_220F040(v27, (__int64)v18, v26, v19);
            ++*(_QWORD *)(a2 + 568);
          }
          if ( v9 == ++v15 )
            break;
        }
        goto LABEL_27;
      }
    }
    si128 = _mm_loadu_si128(a3);
    if ( v7 + 1 > *(unsigned int *)(a2 + 12) )
    {
      v29[0] = si128;
      sub_C8D5F0(a2, (const void *)(a2 + 16), v7 + 1, 0x10u, v8, a6);
      si128 = _mm_load_si128(v29);
      v9 = (__m128i *)(*(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8));
    }
    *v9 = si128;
    v23 = *(_QWORD *)a2;
    v24 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
    *(_DWORD *)(a2 + 8) = v24;
    *(_BYTE *)(a1 + 8) = 1;
    *(_QWORD *)a1 = v23 + 16 * v24 - 16;
    *(_BYTE *)(a1 + 16) = 1;
  }
  return a1;
}
