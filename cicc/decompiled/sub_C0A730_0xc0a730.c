// Function: sub_C0A730
// Address: 0xc0a730
//
__int64 __fastcall sub_C0A730(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi
  unsigned int v5; // r13d
  __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 result; // rax
  const void *v11; // rax
  const void *v12; // rsi
  size_t v13; // rdx
  int v14; // eax

  v3 = (_QWORD *)(a1 + 24);
  *(v3 - 3) = *(_QWORD *)a2;
  *(_QWORD *)(a1 + 8) = v3;
  *(_QWORD *)(a1 + 16) = 0x800000000LL;
  v5 = *(_DWORD *)(a2 + 16);
  if ( v5 && a1 + 8 != a2 + 8 )
  {
    v11 = *(const void **)(a2 + 8);
    v12 = (const void *)(a2 + 24);
    if ( v11 == v12 )
    {
      v13 = 16LL * v5;
      if ( v5 <= 8
        || (sub_C8D5F0(a1 + 8, v3, v5, 16),
            v3 = *(_QWORD **)(a1 + 8),
            v12 = *(const void **)(a2 + 8),
            (v13 = 16LL * *(unsigned int *)(a2 + 16)) != 0) )
      {
        memcpy(v3, v12, v13);
      }
      *(_DWORD *)(a1 + 16) = v5;
      *(_DWORD *)(a2 + 16) = 0;
    }
    else
    {
      *(_QWORD *)(a1 + 8) = v11;
      v14 = *(_DWORD *)(a2 + 20);
      *(_DWORD *)(a1 + 16) = v5;
      *(_DWORD *)(a1 + 20) = v14;
      *(_QWORD *)(a2 + 8) = v12;
      *(_QWORD *)(a2 + 16) = 0;
    }
  }
  *(_QWORD *)(a1 + 152) = a1 + 168;
  v6 = *(_QWORD *)(a2 + 152);
  if ( v6 == a2 + 168 )
  {
    *(__m128i *)(a1 + 168) = _mm_loadu_si128((const __m128i *)(a2 + 168));
  }
  else
  {
    *(_QWORD *)(a1 + 152) = v6;
    *(_QWORD *)(a1 + 168) = *(_QWORD *)(a2 + 168);
  }
  v7 = *(_QWORD *)(a2 + 160);
  *(_QWORD *)(a2 + 152) = a2 + 168;
  *(_QWORD *)(a2 + 160) = 0;
  *(_QWORD *)(a1 + 160) = v7;
  *(_BYTE *)(a2 + 168) = 0;
  *(_QWORD *)(a1 + 184) = a1 + 200;
  v8 = *(_QWORD *)(a2 + 184);
  if ( v8 == a2 + 200 )
  {
    *(__m128i *)(a1 + 200) = _mm_loadu_si128((const __m128i *)(a2 + 200));
  }
  else
  {
    *(_QWORD *)(a1 + 184) = v8;
    *(_QWORD *)(a1 + 200) = *(_QWORD *)(a2 + 200);
  }
  v9 = *(_QWORD *)(a2 + 192);
  *(_QWORD *)(a2 + 184) = a2 + 200;
  *(_QWORD *)(a2 + 192) = 0;
  *(_QWORD *)(a1 + 192) = v9;
  result = *(unsigned int *)(a2 + 216);
  *(_BYTE *)(a2 + 200) = 0;
  *(_DWORD *)(a1 + 216) = result;
  return result;
}
