// Function: sub_D19E40
// Address: 0xd19e40
//
__int64 *__fastcall sub_D19E40(__int64 a1, __int64 a2)
{
  __m128i *v3; // rdx
  __m128i si128; // xmm0
  __int64 v5; // r12
  const char *v6; // rax
  size_t v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  _BYTE *v11; // rdi
  unsigned __int8 *v12; // rsi
  unsigned __int64 v13; // rax
  __int64 v14; // rsi
  __int64 *result; // rax
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  __int64 *v18; // rbx
  __m128i *v19; // rdx
  __int64 v20; // r12
  unsigned int v21; // r14d
  __int64 v22; // rax
  __int64 v23; // rdx
  _BYTE *v24; // rax
  __int64 v25; // rax
  __int64 *v26; // r12
  __int64 v27; // rax
  __int64 v28; // rdx
  _DWORD *v29; // rdx
  unsigned __int8 *v30; // r14
  __m128i *v31; // rdx
  __int64 v32; // r13
  unsigned int v33; // edx
  unsigned int v34; // r14d
  __int64 *v36; // [rsp+18h] [rbp-A8h]
  __int64 v37; // [rsp+20h] [rbp-A0h]
  unsigned int v38; // [rsp+2Ch] [rbp-94h]
  __int64 *v39; // [rsp+30h] [rbp-90h]
  size_t v40; // [rsp+38h] [rbp-88h]
  __int64 v41; // [rsp+48h] [rbp-78h] BYREF
  __int64 *v42; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v43; // [rsp+58h] [rbp-68h]
  _QWORD v44[2]; // [rsp+60h] [rbp-60h] BYREF
  __int64 v45; // [rsp+70h] [rbp-50h]
  __int16 v46; // [rsp+80h] [rbp-40h]

  v3 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v3 <= 0x38u )
  {
    v5 = sub_CB6200(a2, "Printing analysis 'Demanded Bits Analysis' for function '", 0x39u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F70AE0);
    v3[3].m128i_i8[8] = 39;
    v5 = a2;
    v3[3].m128i_i64[0] = 0x206E6F6974636E75LL;
    *v3 = si128;
    v3[1] = _mm_load_si128((const __m128i *)&xmmword_3F70AF0);
    v3[2] = _mm_load_si128((const __m128i *)&xmmword_3F70B00);
    *(_QWORD *)(a2 + 32) += 57LL;
  }
  v6 = sub_BD5D20(*(_QWORD *)a1);
  v11 = *(_BYTE **)(v5 + 32);
  v12 = (unsigned __int8 *)v6;
  v13 = *(_QWORD *)(v5 + 24) - (_QWORD)v11;
  if ( v7 > v13 )
  {
    v17 = sub_CB6200(v5, v12, v7);
    v11 = *(_BYTE **)(v17 + 32);
    v5 = v17;
    v13 = *(_QWORD *)(v17 + 24) - (_QWORD)v11;
  }
  else if ( v7 )
  {
    v40 = v7;
    memcpy(v11, v12, v7);
    v7 = v40;
    v11 = (_BYTE *)(v40 + *(_QWORD *)(v5 + 32));
    v16 = *(_QWORD *)(v5 + 24) - (_QWORD)v11;
    *(_QWORD *)(v5 + 32) = v11;
    if ( v16 > 2 )
      goto LABEL_6;
LABEL_10:
    v14 = (__int64)"':\n";
    sub_CB6200(v5, "':\n", 3u);
    goto LABEL_7;
  }
  if ( v13 <= 2 )
    goto LABEL_10;
LABEL_6:
  v14 = 14887;
  v11[2] = 10;
  *(_WORD *)v11 = 14887;
  *(_QWORD *)(v5 + 32) += 3LL;
LABEL_7:
  result = (__int64 *)sub_D19710(a1, v14, v7, v8, v9, v10);
  if ( *(_DWORD *)(a1 + 336) )
  {
    result = *(__int64 **)(a1 + 328);
    v39 = &result[3 * *(unsigned int *)(a1 + 344)];
    if ( result != v39 )
    {
      while ( 1 )
      {
        v18 = result;
        v37 = *result;
        if ( *result != -8192 && *result != -4096 )
          break;
        result += 3;
        if ( v39 == result )
          return result;
      }
      if ( v39 != result )
      {
        while ( 1 )
        {
          v19 = *(__m128i **)(a2 + 32);
          if ( *(_QWORD *)(a2 + 24) - (_QWORD)v19 <= 0xFu )
          {
            v20 = sub_CB6200(a2, "DemandedBits: 0x", 0x10u);
          }
          else
          {
            v20 = a2;
            *v19 = _mm_load_si128((const __m128i *)&xmmword_3F70B10);
            *(_QWORD *)(a2 + 32) += 16LL;
          }
          v21 = *((_DWORD *)v18 + 4);
          if ( v21 > 0x40 )
          {
            v34 = v21 - sub_C444A0((__int64)(v18 + 1));
            v22 = -1;
            if ( v34 <= 0x40 )
              v22 = *(_QWORD *)v18[1];
          }
          else
          {
            v22 = v18[1];
          }
          v42 = (__int64 *)v22;
          v46 = 271;
          v45 = 0;
          v44[0] = &v42;
          sub_CA0E80((__int64)v44, v20);
          v23 = *(_QWORD *)(v20 + 32);
          if ( (unsigned __int64)(*(_QWORD *)(v20 + 24) - v23) <= 4 )
          {
            sub_CB6200(v20, (unsigned __int8 *)" for ", 5u);
          }
          else
          {
            *(_DWORD *)v23 = 1919903264;
            *(_BYTE *)(v23 + 4) = 32;
            *(_QWORD *)(v20 + 32) += 5LL;
          }
          sub_A69870(v37, (_BYTE *)a2, 0);
          v24 = *(_BYTE **)(a2 + 32);
          if ( (unsigned __int64)v24 >= *(_QWORD *)(a2 + 24) )
          {
            sub_CB5D20(a2, 10);
          }
          else
          {
            *(_QWORD *)(a2 + 32) = v24 + 1;
            *v24 = 10;
          }
          v25 = 4LL * (*(_DWORD *)(v37 + 4) & 0x7FFFFFF);
          if ( (*(_BYTE *)(v37 + 7) & 0x40) != 0 )
          {
            v26 = *(__int64 **)(v37 - 8);
            v36 = &v26[v25];
          }
          else
          {
            v36 = (__int64 *)v37;
            v26 = (__int64 *)(v37 - v25 * 8);
          }
          for ( result = &v41; v36 != v26; v26 += 4 )
          {
            v30 = (unsigned __int8 *)*v26;
            sub_D19B10((__int64)&v42, a1, v26);
            v31 = *(__m128i **)(a2 + 32);
            if ( *(_QWORD *)(a2 + 24) - (_QWORD)v31 <= 0xFu )
            {
              v32 = sub_CB6200(a2, "DemandedBits: 0x", 0x10u);
            }
            else
            {
              v32 = a2;
              *v31 = _mm_load_si128((const __m128i *)&xmmword_3F70B10);
              *(_QWORD *)(a2 + 32) += 16LL;
            }
            v38 = v43;
            if ( v43 <= 0x40 )
            {
              v27 = (__int64)v42;
            }
            else
            {
              v33 = v38 - sub_C444A0((__int64)&v42);
              v27 = -1;
              if ( v33 <= 0x40 )
                v27 = *v42;
            }
            v41 = v27;
            v45 = 0;
            v44[0] = &v41;
            v46 = 271;
            sub_CA0E80((__int64)v44, v32);
            v28 = *(_QWORD *)(v32 + 32);
            if ( (unsigned __int64)(*(_QWORD *)(v32 + 24) - v28) <= 4 )
            {
              sub_CB6200(v32, (unsigned __int8 *)" for ", 5u);
            }
            else
            {
              *(_DWORD *)v28 = 1919903264;
              *(_BYTE *)(v28 + 4) = 32;
              *(_QWORD *)(v32 + 32) += 5LL;
            }
            if ( v30 )
            {
              sub_A5BF40(v30, a2, 0, 0);
              v29 = *(_DWORD **)(a2 + 32);
              if ( *(_QWORD *)(a2 + 24) - (_QWORD)v29 <= 3u )
              {
                sub_CB6200(a2, (unsigned __int8 *)" in ", 4u);
              }
              else
              {
                *v29 = 544106784;
                *(_QWORD *)(a2 + 32) += 4LL;
              }
            }
            sub_A69870(v37, (_BYTE *)a2, 0);
            result = *(__int64 **)(a2 + 32);
            if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 24) )
            {
              result = (__int64 *)sub_CB5D20(a2, 10);
            }
            else
            {
              *(_QWORD *)(a2 + 32) = (char *)result + 1;
              *(_BYTE *)result = 10;
            }
            if ( v43 > 0x40 && v42 )
              result = (__int64 *)j_j___libc_free_0_0(v42);
          }
          v18 += 3;
          if ( v18 == v39 )
            break;
          while ( *v18 == -8192 || *v18 == -4096 )
          {
            v18 += 3;
            if ( v39 == v18 )
              return result;
          }
          if ( v39 == v18 )
            break;
          v37 = *v18;
        }
      }
    }
  }
  return result;
}
