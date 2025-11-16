// Function: sub_39101D0
// Address: 0x39101d0
//
__int64 __fastcall sub_39101D0(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        unsigned __int8 *a4,
        size_t a5,
        char a6,
        __int128 a7)
{
  __int64 v8; // r14
  unsigned __int64 v11; // rax
  unsigned __int8 *v12; // r9
  size_t v13; // r8
  __int64 v14; // rcx
  __int64 v15; // r14
  __int64 result; // rax
  __int64 v17; // rdi
  __int64 v18; // rax
  __m128i v19; // xmm0
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  unsigned __int8 *v23; // [rsp+8h] [rbp-88h]
  size_t v24; // [rsp+10h] [rbp-80h]
  int v25; // [rsp+18h] [rbp-78h]
  unsigned __int8 *v26; // [rsp+20h] [rbp-70h] BYREF
  size_t v27; // [rsp+28h] [rbp-68h]
  _QWORD v28[2]; // [rsp+40h] [rbp-50h] BYREF
  int v29; // [rsp+50h] [rbp-40h]

  v8 = a3 - 1;
  sub_390FF80((__int64)&v26, a1, a4, a5);
  v11 = *(unsigned int *)(a1 + 80);
  v12 = v26;
  v13 = v27;
  if ( (unsigned int)v8 < (unsigned int)v11 )
    goto LABEL_2;
  v20 = a3;
  if ( a3 < v11 )
  {
    *(_DWORD *)(a1 + 80) = a3;
    goto LABEL_2;
  }
  if ( a3 <= v11 )
  {
LABEL_2:
    v14 = *(_QWORD *)(a1 + 72);
    goto LABEL_3;
  }
  if ( a3 > (unsigned __int64)*(unsigned int *)(a1 + 84) )
  {
    v23 = v26;
    v24 = v27;
    sub_16CD150(a1 + 72, (const void *)(a1 + 88), a3, 32, v27, (int)v26);
    v11 = *(unsigned int *)(a1 + 80);
    v12 = v23;
    v13 = v24;
    v20 = a3;
  }
  v14 = *(_QWORD *)(a1 + 72);
  v21 = v14 + 32 * v11;
  v22 = v14 + 32 * v20;
  if ( v21 != v22 )
  {
    do
    {
      if ( v21 )
      {
        *(_DWORD *)v21 = 0;
        *(_BYTE *)(v21 + 4) = 0;
        *(_BYTE *)(v21 + 5) = 0;
        *(_QWORD *)(v21 + 8) = 0;
        *(_QWORD *)(v21 + 16) = 0;
        *(_QWORD *)(v21 + 24) = 0;
      }
      v21 += 32;
    }
    while ( v22 != v21 );
    v14 = *(_QWORD *)(a1 + 72);
  }
  *(_DWORD *)(a1 + 80) = a3;
LABEL_3:
  if ( !v13 )
  {
    v12 = "<stdin>";
    v13 = 7;
  }
  v15 = 32 * v8;
  result = 0;
  if ( !*(_BYTE *)(v14 + v15 + 4) )
  {
    sub_390FF80((__int64)v28, a1, v12, v13);
    v17 = *(_QWORD *)(a2 + 8);
    v25 = v29;
    v28[0] = "checksum_offset";
    LOWORD(v29) = 259;
    v18 = sub_38BF8E0(v17, (__int64)v28, 0, 1);
    v19 = _mm_loadu_si128((const __m128i *)&a7);
    *(_DWORD *)(*(_QWORD *)(a1 + 72) + v15) = v25;
    *(_QWORD *)(*(_QWORD *)(a1 + 72) + v15 + 24) = v18;
    *(_BYTE *)(*(_QWORD *)(a1 + 72) + v15 + 4) = 1;
    *(__m128i *)(*(_QWORD *)(a1 + 72) + v15 + 8) = v19;
    *(_BYTE *)(*(_QWORD *)(a1 + 72) + v15 + 5) = a6;
    return 1;
  }
  return result;
}
