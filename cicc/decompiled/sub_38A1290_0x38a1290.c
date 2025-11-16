// Function: sub_38A1290
// Address: 0x38a1290
//
__int64 __fastcall sub_38A1290(__int64 a1, __int64 a2, __int64 *a3, __m128i a4, __m128i si128, double a6)
{
  __int64 result; // rax
  int v9; // eax
  bool v10; // zf
  _BYTE *v11; // rsi
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // rdi
  _BYTE *v15; // rsi
  __int64 v16; // rdx
  _BYTE *v17; // rcx
  __int64 v18; // rax
  __m128 *v19; // rax
  unsigned __int64 v20; // rdx
  int v21; // eax
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // [rsp+8h] [rbp-C8h]
  __int64 v24; // [rsp+10h] [rbp-C0h]
  __int64 v26; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v27; // [rsp+38h] [rbp-98h] BYREF
  _BYTE *v28; // [rsp+40h] [rbp-90h] BYREF
  _BYTE *v29; // [rsp+48h] [rbp-88h]
  _BYTE *v30; // [rsp+50h] [rbp-80h]
  __m128i *v31; // [rsp+60h] [rbp-70h] BYREF
  unsigned __int64 v32; // [rsp+68h] [rbp-68h]
  __m128i v33; // [rsp+70h] [rbp-60h] BYREF
  __m128i *v34; // [rsp+80h] [rbp-50h] BYREF
  unsigned __int64 v35; // [rsp+88h] [rbp-48h]
  __m128i v36[4]; // [rsp+90h] [rbp-40h] BYREF

  result = 0;
  if ( *(_DWORD *)(a1 + 64) == 6 )
  {
    v23 = *(_QWORD *)(a1 + 56);
    v24 = a1 + 8;
    v9 = sub_3887100(a1 + 8);
    *(_DWORD *)(a1 + 64) = v9;
    if ( v9 == 7 )
    {
LABEL_37:
      if ( *(_DWORD *)(a2 + 8) )
      {
        *(_DWORD *)(a1 + 64) = sub_3887100(v24);
        return 0;
      }
      else
      {
        v36[0].m128i_i16[0] = 259;
        v34 = (__m128i *)"operand bundle set must not be empty";
        return sub_38814C0(v24, v23, (__int64)&v34);
      }
    }
    else
    {
      while ( !*(_DWORD *)(a2 + 8) || !(unsigned __int8)sub_388AF10(a1, 4, "expected ',' in input list") )
      {
        v32 = 0;
        v31 = &v33;
        v33.m128i_i8[0] = 0;
        if ( (unsigned __int8)sub_388B0A0(a1, (unsigned __int64 *)&v31)
          || (unsigned __int8)sub_388AF10(a1, 12, "expected '(' in operand bundle") )
        {
          goto LABEL_18;
        }
        v10 = *(_DWORD *)(a1 + 64) == 13;
        v30 = 0;
        v28 = 0;
        v29 = 0;
        if ( !v10 )
        {
          do
          {
            v26 = 0;
            v27 = 0;
            v34 = (__m128i *)"expected type";
            v36[0].m128i_i16[0] = 259;
            if ( (unsigned __int8)sub_3891B00(a1, &v26, (__int64)&v34, 0)
              || (unsigned __int8)sub_38A1070(
                                    (__int64 **)a1,
                                    v26,
                                    &v27,
                                    a3,
                                    *(double *)a4.m128i_i64,
                                    *(double *)si128.m128i_i64,
                                    a6) )
            {
              break;
            }
            v11 = v29;
            if ( v29 == v30 )
            {
              sub_1287830((__int64)&v28, v29, &v27);
            }
            else
            {
              if ( v29 )
              {
                *(_QWORD *)v29 = v27;
                v11 = v29;
              }
              v29 = v11 + 8;
            }
            if ( *(_DWORD *)(a1 + 64) == 13 )
              goto LABEL_24;
          }
          while ( v29 == v28 || !(unsigned __int8)sub_388AF10(a1, 4, "expected ',' in input list") );
          if ( v28 )
            j_j___libc_free_0((unsigned __int64)v28);
LABEL_18:
          if ( v31 != &v33 )
            j_j___libc_free_0((unsigned __int64)v31);
          return 1;
        }
LABEL_24:
        v12 = *(unsigned int *)(a2 + 8);
        if ( (unsigned int)v12 >= *(_DWORD *)(a2 + 12) )
        {
          sub_1740340(a2, 0);
          v12 = *(unsigned int *)(a2 + 8);
        }
        v34 = v36;
        if ( v31 == &v33 )
        {
          a4 = _mm_load_si128(&v33);
          v36[0] = a4;
        }
        else
        {
          v34 = v31;
          v36[0].m128i_i64[0] = v33.m128i_i64[0];
        }
        v13 = v32;
        v32 = 0;
        v14 = (unsigned __int64)v28;
        v15 = v30;
        v33.m128i_i8[0] = 0;
        v35 = v13;
        v16 = 7 * v12;
        v31 = &v33;
        v17 = v29;
        v30 = 0;
        v29 = 0;
        v18 = *(_QWORD *)a2;
        v28 = 0;
        v19 = (__m128 *)(v18 + 8 * v16);
        if ( v19 )
        {
          v19->m128_u64[0] = (unsigned __int64)&v19[1];
          if ( v34 == v36 )
          {
            si128 = _mm_load_si128(v36);
            v19[1] = (__m128)si128;
          }
          else
          {
            v19->m128_u64[0] = (unsigned __int64)v34;
            v19[1].m128_u64[0] = v36[0].m128i_i64[0];
          }
          v20 = v35;
          v19[2].m128_u64[0] = v14;
          v19[2].m128_u64[1] = (unsigned __int64)v17;
          v19->m128_u64[1] = v20;
          v19[3].m128_u64[0] = (unsigned __int64)v15;
        }
        else
        {
          if ( v14 )
            j_j___libc_free_0(v14);
          if ( v34 != v36 )
            j_j___libc_free_0((unsigned __int64)v34);
        }
        ++*(_DWORD *)(a2 + 8);
        v21 = sub_3887100(v24);
        v22 = (unsigned __int64)v28;
        *(_DWORD *)(a1 + 64) = v21;
        if ( v22 )
          j_j___libc_free_0(v22);
        if ( v31 != &v33 )
          j_j___libc_free_0((unsigned __int64)v31);
        if ( *(_DWORD *)(a1 + 64) == 7 )
          goto LABEL_37;
      }
      return 1;
    }
  }
  return result;
}
