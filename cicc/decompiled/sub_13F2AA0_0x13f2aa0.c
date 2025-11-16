// Function: sub_13F2AA0
// Address: 0x13f2aa0
//
__int64 *__fastcall sub_13F2AA0(__int64 *a1, __int64 a2)
{
  __int64 v4; // rdi
  __int64 *result; // rax
  char v6; // dl
  __int64 v7; // r13
  __m128i *v8; // rdx
  __m128i si128; // xmm0
  void *v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // rdi
  __int64 *v14; // rsi
  unsigned int v15; // r8d
  __int64 *v16; // rcx
  int v17; // [rsp+0h] [rbp-50h] BYREF
  __int64 v18; // [rsp+8h] [rbp-48h]
  unsigned int v19; // [rsp+10h] [rbp-40h]
  __int64 v20; // [rsp+18h] [rbp-38h]
  unsigned int v21; // [rsp+20h] [rbp-30h]

  v4 = *a1;
  result = *(__int64 **)(v4 + 8);
  if ( *(__int64 **)(v4 + 16) != result )
    goto LABEL_2;
  v14 = &result[*(unsigned int *)(v4 + 28)];
  v15 = *(_DWORD *)(v4 + 28);
  if ( result != v14 )
  {
    v16 = 0;
    while ( a2 != *result )
    {
      if ( *result == -2 )
        v16 = result;
      if ( v14 == ++result )
      {
        if ( !v16 )
          goto LABEL_27;
        *v16 = a2;
        --*(_DWORD *)(v4 + 32);
        ++*(_QWORD *)v4;
        goto LABEL_6;
      }
    }
    return result;
  }
LABEL_27:
  if ( v15 < *(_DWORD *)(v4 + 24) )
  {
    *(_DWORD *)(v4 + 28) = v15 + 1;
    *v14 = a2;
    ++*(_QWORD *)v4;
  }
  else
  {
LABEL_2:
    result = (__int64 *)sub_16CCBA0(v4, a2);
    if ( !v6 )
      return result;
  }
LABEL_6:
  sub_13F2700(&v17, *(_QWORD *)(a1[1] + 8), *(_QWORD *)a1[2], a2, 0);
  v7 = a1[3];
  v8 = *(__m128i **)(v7 + 24);
  if ( *(_QWORD *)(v7 + 16) - (_QWORD)v8 <= 0x12u )
  {
    v7 = sub_16E7EE0(a1[3], "; LatticeVal for: '", 19);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_428A450);
    v8[1].m128i_i8[2] = 39;
    v8[1].m128i_i16[0] = 8250;
    *v8 = si128;
    *(_QWORD *)(v7 + 24) += 19LL;
  }
  sub_155C2B0(*(_QWORD *)a1[2], v7, 0);
  v10 = *(void **)(v7 + 24);
  if ( *(_QWORD *)(v7 + 16) - (_QWORD)v10 <= 9u )
  {
    sub_16E7EE0(v7, "' in BB: '", 10);
  }
  else
  {
    qmemcpy(v10, "' in BB: '", 10);
    *(_QWORD *)(v7 + 24) += 10LL;
  }
  sub_15537D0(a2, a1[3], 0);
  v11 = a1[3];
  v12 = *(_QWORD *)(v11 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v11 + 16) - v12) <= 5 )
  {
    v11 = sub_16E7EE0(v11, "' is: ", 6);
  }
  else
  {
    *(_DWORD *)v12 = 1936269351;
    *(_WORD *)(v12 + 4) = 8250;
    *(_QWORD *)(v11 + 24) += 6LL;
  }
  v13 = sub_14A8A60(v11, &v17);
  result = *(__int64 **)(v13 + 24);
  if ( *(__int64 **)(v13 + 16) == result )
  {
    result = (__int64 *)sub_16E7EE0(v13, "\n", 1);
    if ( v17 != 3 )
      return result;
  }
  else
  {
    *(_BYTE *)result = 10;
    ++*(_QWORD *)(v13 + 24);
    if ( v17 != 3 )
      return result;
  }
  if ( v21 > 0x40 && v20 )
    result = (__int64 *)j_j___libc_free_0_0(v20);
  if ( v19 > 0x40 )
  {
    if ( v18 )
      return (__int64 *)j_j___libc_free_0_0(v18);
  }
  return result;
}
