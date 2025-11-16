// Function: sub_6EE880
// Address: 0x6ee880
//
void __fastcall sub_6EE880(__int64 a1, _QWORD *a2)
{
  _BOOL4 v2; // ecx
  char v4; // al
  __int64 v5; // r10
  char v6; // al
  __int64 v7; // rdx
  _BOOL8 v8; // r9
  _BOOL8 v9; // r8
  __int64 v10; // rdi
  int v11; // ecx
  __int64 v12; // r10
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 *v18; // rax
  _OWORD v19[4]; // [rsp+0h] [rbp-170h] BYREF
  _OWORD v20[5]; // [rsp+40h] [rbp-130h] BYREF
  __m128i v21; // [rsp+90h] [rbp-E0h]
  __m128i v22; // [rsp+A0h] [rbp-D0h]
  __m128i v23; // [rsp+B0h] [rbp-C0h]
  __m128i v24; // [rsp+C0h] [rbp-B0h]
  __m128i v25; // [rsp+D0h] [rbp-A0h]
  __m128i v26; // [rsp+E0h] [rbp-90h]
  __m128i v27; // [rsp+F0h] [rbp-80h]
  __m128i v28; // [rsp+100h] [rbp-70h]
  __m128i v29; // [rsp+110h] [rbp-60h]
  __m128i v30; // [rsp+120h] [rbp-50h]
  __m128i v31; // [rsp+130h] [rbp-40h]
  __m128i v32; // [rsp+140h] [rbp-30h]
  __m128i v33; // [rsp+150h] [rbp-20h]

  v2 = 0;
  if ( a2 )
    v2 = (*(_BYTE *)(a1 + 18) & 0x28) == 8;
  v19[0] = _mm_loadu_si128((const __m128i *)a1);
  v4 = *(_BYTE *)(a1 + 16);
  v19[1] = _mm_loadu_si128((const __m128i *)(a1 + 16));
  v19[2] = _mm_loadu_si128((const __m128i *)(a1 + 32));
  v19[3] = _mm_loadu_si128((const __m128i *)(a1 + 48));
  v20[0] = _mm_loadu_si128((const __m128i *)(a1 + 64));
  v20[1] = _mm_loadu_si128((const __m128i *)(a1 + 80));
  v20[2] = _mm_loadu_si128((const __m128i *)(a1 + 96));
  v20[3] = _mm_loadu_si128((const __m128i *)(a1 + 112));
  v20[4] = _mm_loadu_si128((const __m128i *)(a1 + 128));
  if ( v4 == 2 )
  {
    v21 = _mm_loadu_si128((const __m128i *)(a1 + 144));
    v22 = _mm_loadu_si128((const __m128i *)(a1 + 160));
    v23 = _mm_loadu_si128((const __m128i *)(a1 + 176));
    v24 = _mm_loadu_si128((const __m128i *)(a1 + 192));
    v25 = _mm_loadu_si128((const __m128i *)(a1 + 208));
    v26 = _mm_loadu_si128((const __m128i *)(a1 + 224));
    v27 = _mm_loadu_si128((const __m128i *)(a1 + 240));
    v28 = _mm_loadu_si128((const __m128i *)(a1 + 256));
    v29 = _mm_loadu_si128((const __m128i *)(a1 + 272));
    v30 = _mm_loadu_si128((const __m128i *)(a1 + 288));
    v31 = _mm_loadu_si128((const __m128i *)(a1 + 304));
    v32 = _mm_loadu_si128((const __m128i *)(a1 + 320));
    v33 = _mm_loadu_si128((const __m128i *)(a1 + 336));
  }
  else if ( v4 == 5 || v4 == 1 )
  {
    v21.m128i_i64[0] = *(_QWORD *)(a1 + 144);
  }
  v5 = *(_QWORD *)(a1 + 136);
  v6 = *(_BYTE *)(v5 + 80);
  v7 = v5;
  if ( v6 == 16 )
  {
    v7 = **(_QWORD **)(v5 + 88);
    v6 = *(_BYTE *)(v7 + 80);
  }
  if ( v6 == 24 )
  {
    v7 = *(_QWORD *)(v7 + 88);
    v6 = *(_BYTE *)(v7 + 80);
  }
  if ( v6 == 10 && (v10 = *(_QWORD *)(v7 + 88), (*(_BYTE *)(v10 + 193) & 4) != 0) )
  {
    sub_6DEC10(v10);
    if ( a2 )
      *(_QWORD *)((char *)v20 + 4) = *a2;
    v13 = v12;
    sub_6EA7D0(
      v12,
      v12,
      (_DWORD *)v20 + 1,
      (_OWORD *)((char *)v20 + 12),
      (*(_BYTE *)(a1 + 19) & 1) == 0,
      (*(_BYTE *)(a1 + 18) & 0x40) != 0,
      v11,
      0,
      (_QWORD *)a1);
    v18 = (__int64 *)sub_6ED0D0(a1, v13, v14, v15, v16, v17);
    sub_6E70E0(v18, a1);
  }
  else
  {
    v8 = (*(_BYTE *)(a1 + 18) & 0x40) != 0;
    v9 = (*(_BYTE *)(a1 + 19) & 1) == 0;
    if ( a2 )
      *(_QWORD *)((char *)v20 + 4) = *a2;
    sub_6EA7D0(v5, v5, (_DWORD *)v20 + 1, (_OWORD *)((char *)v20 + 12), v9, v8, v2, 0, (_QWORD *)a1);
  }
  sub_6E4EE0(a1, (__int64)v19);
  sub_6E5820(*(unsigned __int64 **)(a1 + 88), 32);
}
