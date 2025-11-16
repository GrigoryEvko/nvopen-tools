// Function: sub_1BBD9D0
// Address: 0x1bbd9d0
//
void __fastcall sub_1BBD9D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v8; // rdx
  __int64 v9; // rax
  void *v10; // rdi
  __m128i v11; // xmm0
  unsigned int v12; // r13d
  __int64 v13; // rax
  const void *v14; // rsi
  int v15; // eax
  size_t v16; // rdx

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x800000000LL;
  v8 = *(unsigned int *)(a2 + 8);
  if ( (_DWORD)v8 )
    sub_1BB9A40(a1, (char **)a2, v8, a4, a5, a6);
  *(_QWORD *)(a1 + 80) = *(_QWORD *)(a2 + 80);
  *(_BYTE *)(a1 + 88) = *(_BYTE *)(a2 + 88);
  *(_QWORD *)(a1 + 96) = a1 + 112;
  *(_QWORD *)(a1 + 104) = 0x400000000LL;
  if ( *(_DWORD *)(a2 + 104) )
    sub_1BB9DA0(a1 + 96, (char **)(a2 + 96), v8, a4, a5, a6);
  v9 = *(_QWORD *)(a2 + 144);
  v10 = (void *)(a1 + 168);
  v11 = _mm_loadu_si128((const __m128i *)(a2 + 128));
  *(_QWORD *)(a1 + 152) = a1 + 168;
  *(_QWORD *)(a1 + 144) = v9;
  *(_QWORD *)(a1 + 160) = 0x100000000LL;
  *(__m128i *)(a1 + 128) = v11;
  v12 = *(_DWORD *)(a2 + 160);
  if ( v12 && a1 + 152 != a2 + 152 )
  {
    v13 = *(_QWORD *)(a2 + 152);
    v14 = (const void *)(a2 + 168);
    if ( v13 == a2 + 168 )
    {
      v16 = 4;
      if ( v12 == 1
        || (sub_16CD150(a1 + 152, (const void *)(a1 + 168), v12, 4, a1 + 152, v12),
            v10 = *(void **)(a1 + 152),
            v14 = *(const void **)(a2 + 152),
            (v16 = 4LL * *(unsigned int *)(a2 + 160)) != 0) )
      {
        memcpy(v10, v14, v16);
      }
      *(_DWORD *)(a1 + 160) = v12;
      *(_DWORD *)(a2 + 160) = 0;
    }
    else
    {
      *(_QWORD *)(a1 + 152) = v13;
      v15 = *(_DWORD *)(a2 + 164);
      *(_DWORD *)(a1 + 160) = v12;
      *(_DWORD *)(a1 + 164) = v15;
      *(_QWORD *)(a2 + 152) = v14;
      *(_QWORD *)(a2 + 160) = 0;
    }
  }
}
