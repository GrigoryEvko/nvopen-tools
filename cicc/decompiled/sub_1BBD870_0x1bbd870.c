// Function: sub_1BBD870
// Address: 0x1bbd870
//
void __fastcall sub_1BBD870(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v7; // rdx
  __int64 v8; // rax
  void *v9; // rdi
  __m128i v10; // xmm0
  unsigned int v11; // r13d
  size_t v12; // rdx

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x800000000LL;
  v7 = *(unsigned int *)(a2 + 8);
  if ( (_DWORD)v7 )
    sub_1BB9B80(a1, a2, v7, a4, a5, a6);
  *(_QWORD *)(a1 + 80) = *(_QWORD *)(a2 + 80);
  *(_BYTE *)(a1 + 88) = *(_BYTE *)(a2 + 88);
  *(_QWORD *)(a1 + 96) = a1 + 112;
  *(_QWORD *)(a1 + 104) = 0x400000000LL;
  if ( *(_DWORD *)(a2 + 104) )
    sub_1BB9EE0(a1 + 96, a2 + 96, v7, a4, a5, a6);
  v8 = *(_QWORD *)(a2 + 144);
  v9 = (void *)(a1 + 168);
  v10 = _mm_loadu_si128((const __m128i *)(a2 + 128));
  *(_QWORD *)(a1 + 152) = a1 + 168;
  *(_QWORD *)(a1 + 144) = v8;
  *(_QWORD *)(a1 + 160) = 0x100000000LL;
  *(__m128i *)(a1 + 128) = v10;
  v11 = *(_DWORD *)(a2 + 160);
  if ( v11 && a1 + 152 != a2 + 152 )
  {
    v12 = 4;
    if ( v11 == 1
      || (sub_16CD150(a1 + 152, (const void *)(a1 + 168), v11, 4, a1 + 152, v11),
          v9 = *(void **)(a1 + 152),
          (v12 = 4LL * *(unsigned int *)(a2 + 160)) != 0) )
    {
      memcpy(v9, *(const void **)(a2 + 152), v12);
    }
    *(_DWORD *)(a1 + 160) = v11;
  }
}
