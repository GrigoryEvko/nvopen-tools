// Function: sub_234AC00
// Address: 0x234ac00
//
__int64 __fastcall sub_234AC00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  void *v8; // rdi
  unsigned int v9; // r13d
  void *v10; // rdi
  unsigned int v11; // r13d
  void *v12; // rdi
  unsigned int v13; // r13d
  void *v14; // rdi
  unsigned int v15; // r13d
  void *v16; // rdi
  unsigned int v17; // r13d
  const void *v19; // rax
  const void *v20; // rsi
  size_t v21; // rdx
  __int64 v22; // rax
  const void *v23; // rsi
  size_t v24; // rdx
  __int64 v25; // rax
  const void *v26; // rsi
  size_t v27; // rdx
  __int64 v28; // rax
  const void *v29; // rsi
  size_t v30; // rdx
  __int64 v31; // rax
  const void *v32; // rsi
  size_t v33; // rdx
  int v34; // eax
  int v35; // eax
  int v36; // eax
  int v37; // eax
  int v38; // eax

  v8 = (void *)(a1 + 16);
  *(_QWORD *)a1 = v8;
  *(_QWORD *)(a1 + 8) = 0x800000000LL;
  v9 = *(_DWORD *)(a2 + 8);
  if ( v9 && a1 != a2 )
  {
    v19 = *(const void **)a2;
    v20 = (const void *)(a2 + 16);
    if ( v19 == v20 )
    {
      v21 = 8LL * v9;
      if ( v9 <= 8
        || (sub_C8D5F0(a1, v8, v9, 8u, v9, a6),
            v8 = *(void **)a1,
            v20 = *(const void **)a2,
            (v21 = 8LL * *(unsigned int *)(a2 + 8)) != 0) )
      {
        memcpy(v8, v20, v21);
      }
      *(_DWORD *)(a1 + 8) = v9;
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      *(_QWORD *)a1 = v19;
      v38 = *(_DWORD *)(a2 + 12);
      *(_DWORD *)(a1 + 8) = v9;
      *(_DWORD *)(a1 + 12) = v38;
      *(_QWORD *)a2 = v20;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
  v10 = (void *)(a1 + 96);
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_QWORD *)(a1 + 88) = 0x800000000LL;
  v11 = *(_DWORD *)(a2 + 88);
  if ( v11 && a1 + 80 != a2 + 80 )
  {
    v31 = *(_QWORD *)(a2 + 80);
    v32 = (const void *)(a2 + 96);
    if ( v31 == a2 + 96 )
    {
      v33 = 8LL * v11;
      if ( v11 <= 8
        || (sub_C8D5F0(a1 + 80, (const void *)(a1 + 96), v11, 8u, a1 + 80, v11),
            v10 = *(void **)(a1 + 80),
            v32 = *(const void **)(a2 + 80),
            (v33 = 8LL * *(unsigned int *)(a2 + 88)) != 0) )
      {
        memcpy(v10, v32, v33);
      }
      *(_DWORD *)(a1 + 88) = v11;
      *(_DWORD *)(a2 + 88) = 0;
    }
    else
    {
      *(_QWORD *)(a1 + 80) = v31;
      v34 = *(_DWORD *)(a2 + 92);
      *(_DWORD *)(a1 + 88) = v11;
      *(_DWORD *)(a1 + 92) = v34;
      *(_QWORD *)(a2 + 80) = v32;
      *(_QWORD *)(a2 + 88) = 0;
    }
  }
  v12 = (void *)(a1 + 176);
  *(_QWORD *)(a1 + 160) = a1 + 176;
  *(_QWORD *)(a1 + 168) = 0x800000000LL;
  v13 = *(_DWORD *)(a2 + 168);
  if ( v13 && a1 + 160 != a2 + 160 )
  {
    v28 = *(_QWORD *)(a2 + 160);
    v29 = (const void *)(a2 + 176);
    if ( v28 == a2 + 176 )
    {
      v30 = 8LL * v13;
      if ( v13 <= 8
        || (sub_C8D5F0(a1 + 160, (const void *)(a1 + 176), v13, 8u, a1 + 160, v13),
            v12 = *(void **)(a1 + 160),
            v29 = *(const void **)(a2 + 160),
            (v30 = 8LL * *(unsigned int *)(a2 + 168)) != 0) )
      {
        memcpy(v12, v29, v30);
      }
      *(_DWORD *)(a1 + 168) = v13;
      *(_DWORD *)(a2 + 168) = 0;
    }
    else
    {
      *(_QWORD *)(a1 + 160) = v28;
      v35 = *(_DWORD *)(a2 + 172);
      *(_DWORD *)(a1 + 168) = v13;
      *(_DWORD *)(a1 + 172) = v35;
      *(_QWORD *)(a2 + 160) = v29;
      *(_QWORD *)(a2 + 168) = 0;
    }
  }
  v14 = (void *)(a1 + 256);
  *(_QWORD *)(a1 + 240) = a1 + 256;
  *(_QWORD *)(a1 + 248) = 0x800000000LL;
  v15 = *(_DWORD *)(a2 + 248);
  if ( v15 && a1 + 240 != a2 + 240 )
  {
    v25 = *(_QWORD *)(a2 + 240);
    v26 = (const void *)(a2 + 256);
    if ( v25 == a2 + 256 )
    {
      v27 = 8LL * v15;
      if ( v15 <= 8
        || (sub_C8D5F0(a1 + 240, (const void *)(a1 + 256), v15, 8u, a1 + 240, v15),
            v14 = *(void **)(a1 + 240),
            v26 = *(const void **)(a2 + 240),
            (v27 = 8LL * *(unsigned int *)(a2 + 248)) != 0) )
      {
        memcpy(v14, v26, v27);
      }
      *(_DWORD *)(a1 + 248) = v15;
      *(_DWORD *)(a2 + 248) = 0;
    }
    else
    {
      *(_QWORD *)(a1 + 240) = v25;
      v36 = *(_DWORD *)(a2 + 252);
      *(_DWORD *)(a1 + 248) = v15;
      *(_DWORD *)(a1 + 252) = v36;
      *(_QWORD *)(a2 + 240) = v26;
      *(_QWORD *)(a2 + 248) = 0;
    }
  }
  v16 = (void *)(a1 + 336);
  *(_QWORD *)(a1 + 320) = a1 + 336;
  *(_QWORD *)(a1 + 328) = 0x800000000LL;
  v17 = *(_DWORD *)(a2 + 328);
  if ( v17 && a1 + 320 != a2 + 320 )
  {
    v22 = *(_QWORD *)(a2 + 320);
    v23 = (const void *)(a2 + 336);
    if ( v22 == a2 + 336 )
    {
      v24 = 8LL * v17;
      if ( v17 <= 8
        || (sub_C8D5F0(a1 + 320, (const void *)(a1 + 336), v17, 8u, a1 + 320, v17),
            v16 = *(void **)(a1 + 320),
            v23 = *(const void **)(a2 + 320),
            (v24 = 8LL * *(unsigned int *)(a2 + 328)) != 0) )
      {
        memcpy(v16, v23, v24);
      }
      *(_DWORD *)(a1 + 328) = v17;
      *(_DWORD *)(a2 + 328) = 0;
    }
    else
    {
      *(_QWORD *)(a1 + 320) = v22;
      v37 = *(_DWORD *)(a2 + 332);
      *(_DWORD *)(a1 + 328) = v17;
      *(_DWORD *)(a1 + 332) = v37;
      *(_QWORD *)(a2 + 320) = v23;
      *(_QWORD *)(a2 + 328) = 0;
    }
  }
  return sub_C8CF70(a1 + 400, (void *)(a1 + 432), 32, a2 + 432, a2 + 400);
}
