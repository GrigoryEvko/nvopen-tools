// Function: sub_13F2D10
// Address: 0x13f2d10
//
int *__fastcall sub_13F2D10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r13
  __int64 v6; // rbx
  int *result; // rax
  __int64 v8; // rcx
  __m128i *v9; // rdx
  __int64 v10; // r8
  __m128i si128; // xmm0
  __int64 v12; // r8
  __int64 v13; // rdx
  __int64 v14; // rdi
  int *v15; // [rsp+0h] [rbp-70h]
  __int64 v16; // [rsp+8h] [rbp-68h]
  int v17; // [rsp+10h] [rbp-60h] BYREF
  __int64 v18; // [rsp+18h] [rbp-58h]
  unsigned int v19; // [rsp+20h] [rbp-50h]
  __int64 v20; // [rsp+28h] [rbp-48h]
  unsigned int v21; // [rsp+30h] [rbp-40h]

  v5 = *(_QWORD *)(a2 + 56);
  if ( (*(_BYTE *)(v5 + 18) & 1) != 0 )
  {
    sub_15E08E0(*(_QWORD *)(a2 + 56));
    v6 = *(_QWORD *)(v5 + 88);
    result = (int *)(v6 + 40LL * *(_QWORD *)(v5 + 96));
    v15 = result;
    if ( (*(_BYTE *)(v5 + 18) & 1) != 0 )
    {
      result = (int *)sub_15E08E0(v5);
      v6 = *(_QWORD *)(v5 + 88);
    }
  }
  else
  {
    v6 = *(_QWORD *)(v5 + 88);
    result = (int *)(v6 + 40LL * *(_QWORD *)(v5 + 96));
    v15 = result;
  }
  if ( (int *)v6 != v15 )
  {
    while ( 1 )
    {
      result = sub_13F2700(&v17, *(_QWORD *)(a1 + 8), v6, a2, 0);
      if ( !v17 )
        goto LABEL_6;
      v9 = *(__m128i **)(a3 + 24);
      v10 = a3;
      if ( *(_QWORD *)(a3 + 16) - (_QWORD)v9 <= 0x12u )
      {
        v10 = sub_16E7EE0(a3, "; LatticeVal for: '", 19, v8, a3);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_428A450);
        v9[1].m128i_i8[2] = 39;
        v9[1].m128i_i16[0] = 8250;
        *v9 = si128;
        *(_QWORD *)(a3 + 24) += 19LL;
      }
      v16 = v10;
      sub_155C2B0(v6, v10, 0);
      v12 = v16;
      v13 = *(_QWORD *)(v16 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(v16 + 16) - v13) <= 5 )
      {
        v12 = sub_16E7EE0(v16, "' is: ", 6);
      }
      else
      {
        *(_DWORD *)v13 = 1936269351;
        *(_WORD *)(v13 + 4) = 8250;
        *(_QWORD *)(v16 + 24) += 6LL;
      }
      v14 = sub_14A8A60(v12, &v17);
      result = *(int **)(v14 + 24);
      if ( *(int **)(v14 + 16) == result )
      {
        result = (int *)sub_16E7EE0(v14, "\n", 1);
        if ( v17 == 3 )
          goto LABEL_14;
LABEL_6:
        v6 += 40;
        if ( v15 == (int *)v6 )
          return result;
      }
      else
      {
        *(_BYTE *)result = 10;
        ++*(_QWORD *)(v14 + 24);
        if ( v17 != 3 )
          goto LABEL_6;
LABEL_14:
        if ( v21 > 0x40 && v20 )
          result = (int *)j_j___libc_free_0_0(v20);
        if ( v19 <= 0x40 || !v18 )
          goto LABEL_6;
        result = (int *)j_j___libc_free_0_0(v18);
        v6 += 40;
        if ( v15 == (int *)v6 )
          return result;
      }
    }
  }
  return result;
}
