// Function: sub_125BB70
// Address: 0x125bb70
//
__int64 __fastcall sub_125BB70(__int64 a1, __int64 a2)
{
  _BYTE *v3; // r12
  __int64 result; // rax
  _BYTE *v5; // rdi
  size_t v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // r8
  size_t v9; // rdx
  void *src; // [rsp+0h] [rbp-40h] BYREF
  size_t n; // [rsp+8h] [rbp-38h]
  _QWORD v12[6]; // [rsp+10h] [rbp-30h] BYREF

  v3 = (_BYTE *)(a2 + 16);
  *(_WORD *)a1 = 6;
  if ( (unsigned __int8)sub_C6A630(*(char **)a2, *(_QWORD *)(a2 + 8), 0) )
    goto LABEL_2;
  sub_C6B0E0((__int64 *)&src, *(_QWORD *)a2, *(_QWORD *)(a2 + 8));
  v5 = *(_BYTE **)a2;
  if ( src == v12 )
  {
    v9 = n;
    if ( n )
    {
      if ( n == 1 )
        *v5 = v12[0];
      else
        memcpy(v5, src, n);
      v9 = n;
      v5 = *(_BYTE **)a2;
    }
    *(_QWORD *)(a2 + 8) = v9;
    v5[v9] = 0;
    v5 = src;
    goto LABEL_10;
  }
  v6 = n;
  v7 = v12[0];
  if ( v5 == v3 )
  {
    *(_QWORD *)a2 = src;
    *(_QWORD *)(a2 + 8) = v6;
    *(_QWORD *)(a2 + 16) = v7;
    goto LABEL_12;
  }
  v8 = *(_QWORD *)(a2 + 16);
  *(_QWORD *)a2 = src;
  *(_QWORD *)(a2 + 8) = v6;
  *(_QWORD *)(a2 + 16) = v7;
  if ( !v5 )
  {
LABEL_12:
    src = v12;
    v5 = v12;
    goto LABEL_10;
  }
  src = v5;
  v12[0] = v8;
LABEL_10:
  n = 0;
  *v5 = 0;
  sub_2240A30(&src);
LABEL_2:
  *(_QWORD *)(a1 + 8) = a1 + 24;
  if ( *(_BYTE **)a2 == v3 )
  {
    *(__m128i *)(a1 + 24) = _mm_loadu_si128((const __m128i *)(a2 + 16));
  }
  else
  {
    *(_QWORD *)(a1 + 8) = *(_QWORD *)a2;
    *(_QWORD *)(a1 + 24) = *(_QWORD *)(a2 + 16);
  }
  result = *(_QWORD *)(a2 + 8);
  *(_QWORD *)a2 = v3;
  *(_QWORD *)(a2 + 8) = 0;
  *(_QWORD *)(a1 + 16) = result;
  *(_BYTE *)(a2 + 16) = 0;
  return result;
}
