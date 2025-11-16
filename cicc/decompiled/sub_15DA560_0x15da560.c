// Function: sub_15DA560
// Address: 0x15da560
//
__int64 __fastcall sub_15DA560(const __m128i *a1, __int64 *a2, unsigned __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 *v5; // r15
  __int64 v7; // rbx
  __int64 i; // r14
  __int64 v9; // rax
  char v10; // al
  __int64 *v11; // rdx
  __int64 v12; // rax
  char v13; // al
  __int64 *v14; // rcx
  unsigned int v15; // eax
  int v16; // eax
  unsigned int v17; // esi
  __int64 v18; // rax
  unsigned int v19; // eax
  int v20; // eax
  unsigned int v21; // esi
  unsigned __int64 v22; // rax
  __int64 *v23; // rcx
  __int64 v24; // r8
  __int64 v25; // [rsp+0h] [rbp-80h]
  int v27; // [rsp+18h] [rbp-68h]
  __int64 *v28; // [rsp+28h] [rbp-58h] BYREF
  __int64 v29; // [rsp+30h] [rbp-50h] BYREF
  unsigned __int64 v30; // [rsp+38h] [rbp-48h]
  __int64 *v31; // [rsp+40h] [rbp-40h] BYREF
  unsigned __int64 v32; // [rsp+48h] [rbp-38h]

  result = (char *)a2 - (char *)a1;
  v5 = a2;
  if ( (char *)a2 - (char *)a1 > 16 )
  {
    v7 = result >> 4;
    for ( i = ((result >> 4) - 2) / 2; ; --i )
    {
      result = sub_15D9E50((__int64)a1, i, v7, (__int64 *)a1[i].m128i_i64[0], a1[i].m128i_i64[1], a4);
      if ( !i )
        break;
    }
  }
  if ( (unsigned __int64)a2 < a3 )
  {
    v25 = ((char *)a2 - (char *)a1) >> 4;
    while ( 1 )
    {
      v12 = v5[1];
      v29 = *v5;
      v30 = v12 & 0xFFFFFFFFFFFFFFF8LL;
      v13 = sub_15D0A10(a4, &v29, &v31);
      v14 = v31;
      if ( !v13 )
        break;
      v27 = *((_DWORD *)v31 + 4);
LABEL_8:
      v9 = a1->m128i_i64[1];
      v31 = (__int64 *)a1->m128i_i64[0];
      v32 = v9 & 0xFFFFFFFFFFFFFFF8LL;
      v10 = sub_15D0A10(a4, (__int64 *)&v31, &v28);
      v11 = v28;
      if ( v10 )
      {
        result = *((unsigned int *)v28 + 4);
        if ( v27 > (int)result )
          goto LABEL_24;
LABEL_10:
        v5 += 2;
        if ( a3 <= (unsigned __int64)v5 )
          return result;
      }
      else
      {
        v19 = *(_DWORD *)(a4 + 8);
        ++*(_QWORD *)a4;
        v20 = (v19 >> 1) + 1;
        if ( (*(_BYTE *)(a4 + 8) & 1) != 0 )
        {
          v21 = 4;
          if ( (unsigned int)(4 * v20) >= 0xC )
          {
LABEL_30:
            v21 *= 2;
LABEL_31:
            sub_15D0B40(a4, v21);
            sub_15D0A10(a4, (__int64 *)&v31, &v28);
            v11 = v28;
            v20 = (*(_DWORD *)(a4 + 8) >> 1) + 1;
            goto LABEL_21;
          }
        }
        else
        {
          v21 = *(_DWORD *)(a4 + 24);
          if ( 4 * v20 >= 3 * v21 )
            goto LABEL_30;
        }
        if ( v21 - (v20 + *(_DWORD *)(a4 + 12)) <= v21 >> 3 )
          goto LABEL_31;
LABEL_21:
        *(_DWORD *)(a4 + 8) = *(_DWORD *)(a4 + 8) & 1 | (2 * v20);
        if ( *v11 != -8 || v11[1] != -8 )
          --*(_DWORD *)(a4 + 12);
        *v11 = (__int64)v31;
        v22 = v32;
        *((_DWORD *)v11 + 4) = 0;
        v11[1] = v22;
        result = 0;
        if ( v27 <= 0 )
          goto LABEL_10;
LABEL_24:
        v23 = (__int64 *)*v5;
        v24 = v5[1];
        v5 += 2;
        *((__m128i *)v5 - 1) = _mm_loadu_si128(a1);
        result = sub_15D9E50((__int64)a1, 0, v25, v23, v24, a4);
        if ( a3 <= (unsigned __int64)v5 )
          return result;
      }
    }
    v15 = *(_DWORD *)(a4 + 8);
    ++*(_QWORD *)a4;
    v16 = (v15 >> 1) + 1;
    if ( (*(_BYTE *)(a4 + 8) & 1) != 0 )
    {
      v17 = 4;
      if ( (unsigned int)(4 * v16) < 0xC )
      {
LABEL_14:
        if ( v17 - (v16 + *(_DWORD *)(a4 + 12)) > v17 >> 3 )
          goto LABEL_15;
        goto LABEL_28;
      }
    }
    else
    {
      v17 = *(_DWORD *)(a4 + 24);
      if ( 3 * v17 > 4 * v16 )
        goto LABEL_14;
    }
    v17 *= 2;
LABEL_28:
    sub_15D0B40(a4, v17);
    sub_15D0A10(a4, &v29, &v31);
    v14 = v31;
    v16 = (*(_DWORD *)(a4 + 8) >> 1) + 1;
LABEL_15:
    *(_DWORD *)(a4 + 8) = *(_DWORD *)(a4 + 8) & 1 | (2 * v16);
    if ( *v14 != -8 || v14[1] != -8 )
      --*(_DWORD *)(a4 + 12);
    v18 = v29;
    *((_DWORD *)v14 + 4) = 0;
    v27 = 0;
    *v14 = v18;
    v14[1] = v30;
    goto LABEL_8;
  }
  return result;
}
