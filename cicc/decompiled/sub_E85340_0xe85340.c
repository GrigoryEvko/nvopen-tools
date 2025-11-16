// Function: sub_E85340
// Address: 0xe85340
//
__int64 __fastcall sub_E85340(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5, _QWORD *a6)
{
  __int64 v8; // rcx
  unsigned __int64 v11; // rax
  int v13; // edx
  __int64 v14; // rsi
  int v15; // edi
  __int64 v16; // r9
  __int64 v17; // r8
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // rcx
  const __m128i *v20; // rbx
  __m128i *v21; // rax
  __int64 v23; // rdi
  __int64 v24; // rdi
  __int64 v25; // rdi
  __int64 v26; // r8
  char *v27; // rbx
  unsigned __int8 v28; // [rsp+Fh] [rbp-61h]
  unsigned __int8 v29; // [rsp+Fh] [rbp-61h]
  const char *v30; // [rsp+10h] [rbp-60h] BYREF
  int v31; // [rsp+18h] [rbp-58h]
  __int64 v32; // [rsp+20h] [rbp-50h]
  int v33; // [rsp+28h] [rbp-48h]
  char v34; // [rsp+30h] [rbp-40h]
  char v35; // [rsp+31h] [rbp-3Fh]

  if ( (*(_BYTE *)(a2 + 48) & 0x20) == 0 )
  {
    v24 = *(_QWORD *)(a1 + 8);
    v35 = 1;
    v34 = 3;
    v30 = "The usage of .zerofill is restricted to sections of ZEROFILL type. Use .zero or .space instead.";
    return sub_E66880(v24, a6, (__int64)&v30);
  }
  v8 = *(unsigned int *)(a1 + 128);
  v11 = *(_QWORD *)(a1 + 120);
  v13 = *(_DWORD *)(a1 + 128);
  v14 = 32 * v8;
  if ( (_DWORD)v8 )
  {
    v23 = v11 + v14 - 32;
    v17 = *(_QWORD *)(v23 + 16);
    v13 = *(_DWORD *)(v23 + 24);
    v16 = *(_QWORD *)v23;
    v15 = *(_DWORD *)(v23 + 8);
  }
  else
  {
    v15 = 0;
    v16 = 0;
    v17 = 0;
  }
  v33 = v13;
  v18 = v8 + 1;
  v19 = *(unsigned int *)(a1 + 132);
  v20 = (const __m128i *)&v30;
  v30 = (const char *)v16;
  v31 = v15;
  v32 = v17;
  if ( v18 > v19 )
  {
    v25 = a1 + 120;
    v26 = a1 + 136;
    if ( v11 > (unsigned __int64)&v30 )
    {
      v29 = a5;
    }
    else
    {
      v29 = a5;
      if ( (unsigned __int64)&v30 < v11 + v14 )
      {
        v27 = (char *)&v30 - v11;
        sub_C8D5F0(v25, (const void *)(a1 + 136), v18, 0x20u, v26, v16);
        v11 = *(_QWORD *)(a1 + 120);
        a5 = v29;
        v20 = (const __m128i *)&v27[v11];
        v14 = 32LL * *(unsigned int *)(a1 + 128);
        goto LABEL_5;
      }
    }
    sub_C8D5F0(v25, (const void *)(a1 + 136), v18, 0x20u, v26, v16);
    v11 = *(_QWORD *)(a1 + 120);
    a5 = v29;
    v14 = 32LL * *(unsigned int *)(a1 + 128);
  }
LABEL_5:
  v21 = (__m128i *)(v14 + v11);
  v28 = a5;
  *v21 = _mm_loadu_si128(v20);
  v21[1] = _mm_loadu_si128(v20 + 1);
  ++*(_DWORD *)(a1 + 128);
  sub_E980F0(a1, a2, 0);
  if ( a3 )
  {
    sub_E8B560(a1, v28, 0, 1, 0);
    sub_E85210(a1, a3, 0);
    sub_E99300(a1, a4);
  }
  return sub_E97E50(a1);
}
