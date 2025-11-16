// Function: sub_1680880
// Address: 0x1680880
//
__int64 __fastcall sub_1680880(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r12d
  __m128i v4; // xmm0
  char v5; // al
  __int64 v6; // rcx
  unsigned int v8; // esi
  int v9; // eax
  int v10; // eax
  unsigned __int64 v11; // rax
  __int64 v12; // [rsp+8h] [rbp-68h] BYREF
  __m128i v13; // [rsp+10h] [rbp-60h] BYREF
  __m128i v14; // [rsp+30h] [rbp-40h] BYREF
  __int64 v15; // [rsp+40h] [rbp-30h]

  v3 = a3;
  v13.m128i_i64[0] = a2;
  v13.m128i_i64[1] = a3;
  v4 = _mm_loadu_si128(&v13);
  v15 = 0;
  v14 = v4;
  v5 = sub_167FF60(a1, (char **)&v14, &v12);
  v6 = v12;
  if ( v5 )
    return *(_QWORD *)(v6 + 16);
  v8 = *(_DWORD *)(a1 + 24);
  v9 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v10 = v9 + 1;
  if ( 4 * v10 >= 3 * v8 )
  {
    v8 *= 2;
  }
  else if ( v8 - *(_DWORD *)(a1 + 20) - v10 > v8 >> 3 )
  {
    goto LABEL_5;
  }
  sub_16805B0(a1, v8);
  sub_167FF60(a1, (char **)&v14, &v12);
  v6 = v12;
  v10 = *(_DWORD *)(a1 + 16) + 1;
LABEL_5:
  *(_DWORD *)(a1 + 16) = v10;
  if ( *(_DWORD *)(v6 + 12) || *(_QWORD *)v6 != -1 )
    --*(_DWORD *)(a1 + 20);
  *(__m128i *)v6 = _mm_loadu_si128(&v14);
  *(_QWORD *)(v6 + 16) = v15;
  v11 = *(unsigned int *)(a1 + 44)
      * (((unsigned __int64)*(unsigned int *)(a1 + 44) + *(_QWORD *)(a1 + 32) - 1)
       / *(unsigned int *)(a1 + 44));
  *(_QWORD *)(v6 + 16) = v11;
  *(_QWORD *)(a1 + 32) = (*(_DWORD *)(a1 + 40) != 3) + (unsigned __int64)v3 + v11;
  return *(_QWORD *)(v6 + 16);
}
