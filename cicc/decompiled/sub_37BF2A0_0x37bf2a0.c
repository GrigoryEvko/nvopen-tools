// Function: sub_37BF2A0
// Address: 0x37bf2a0
//
__int64 __fastcall sub_37BF2A0(__int64 a1, __int64 a2, int *a3, __int64 a4)
{
  __int64 v8; // rsi
  __int64 v9; // r8
  int v10; // eax
  __int64 v11; // rdi
  int v12; // r11d
  int *v13; // r15
  unsigned int v14; // edx
  unsigned int *v15; // rcx
  __int64 v16; // r9
  __int64 v18; // r8
  __int64 v19; // rdx
  __int64 v20; // rcx
  __m128i v21; // xmm0
  __int64 v22; // rax
  __int64 v23; // rdx
  int *v24; // [rsp+18h] [rbp-38h] BYREF

  v8 = *(unsigned int *)(a2 + 24);
  v9 = *(_QWORD *)a2;
  if ( !(_DWORD)v8 )
  {
    v24 = 0;
    *(_QWORD *)a2 = v9 + 1;
LABEL_21:
    LODWORD(v8) = 2 * v8;
    goto LABEL_22;
  }
  v10 = *a3;
  v11 = *(_QWORD *)(a2 + 8);
  v12 = 1;
  v13 = 0;
  v14 = (v8 - 1) & (37 * *a3);
  v15 = (unsigned int *)(v11 + 88LL * v14);
  v16 = *v15;
  if ( v10 == (_DWORD)v16 )
  {
LABEL_3:
    *(_QWORD *)a1 = a2;
    *(_QWORD *)(a1 + 8) = v9;
    *(_QWORD *)(a1 + 16) = v15;
    *(_QWORD *)(a1 + 24) = v11 + 88 * v8;
    *(_BYTE *)(a1 + 32) = 0;
    return a1;
  }
  while ( (_DWORD)v16 != -1 )
  {
    if ( (_DWORD)v16 == -2 && !v13 )
      v13 = (int *)v15;
    v14 = (v8 - 1) & (v12 + v14);
    v15 = (unsigned int *)(v11 + 88LL * v14);
    v16 = *v15;
    if ( v10 == (_DWORD)v16 )
      goto LABEL_3;
    ++v12;
  }
  if ( !v13 )
    v13 = (int *)v15;
  v18 = v9 + 1;
  v19 = (unsigned int)(*(_DWORD *)(a2 + 16) + 1);
  *(_QWORD *)a2 = v18;
  v24 = v13;
  if ( 4 * (int)v19 >= (unsigned int)(3 * v8) )
    goto LABEL_21;
  v20 = (unsigned int)v8 >> 3;
  if ( (int)v8 - *(_DWORD *)(a2 + 20) - (int)v19 <= (unsigned int)v20 )
  {
LABEL_22:
    sub_37BF080(a2, v8);
    sub_37BEB80(a2, a3, &v24);
    v13 = v24;
    v19 = (unsigned int)(*(_DWORD *)(a2 + 16) + 1);
  }
  *(_DWORD *)(a2 + 16) = v19;
  if ( *v13 != -1 )
    --*(_DWORD *)(a2 + 20);
  *v13 = *a3;
  *((_QWORD *)v13 + 1) = v13 + 6;
  *((_QWORD *)v13 + 2) = 0x100000000LL;
  if ( *(_DWORD *)(a4 + 8) )
    sub_37B6900((__int64)(v13 + 2), (char **)a4, v19, v20, v18, v16);
  v21 = _mm_loadu_si128((const __m128i *)(a4 + 64));
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = v13;
  *(__m128i *)(v13 + 18) = v21;
  v22 = *(unsigned int *)(a2 + 24);
  *(_BYTE *)(a1 + 32) = 1;
  v23 = *(_QWORD *)a2;
  *(_QWORD *)(a1 + 24) = *(_QWORD *)(a2 + 8) + 88 * v22;
  *(_QWORD *)(a1 + 8) = v23;
  return a1;
}
