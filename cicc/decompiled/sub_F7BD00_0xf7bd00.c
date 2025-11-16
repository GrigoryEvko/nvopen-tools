// Function: sub_F7BD00
// Address: 0xf7bd00
//
__int64 __fastcall sub_F7BD00(unsigned __int64 **a1, unsigned int a2, unsigned int a3, unsigned int a4)
{
  __int64 v5; // rsi
  __int64 v8; // r13
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // r8
  unsigned __int64 v12; // rcx
  int v13; // eax
  __int64 *v14; // r15
  __int64 v15; // r13
  _QWORD *v16; // rdi
  int v17; // edx
  int v18; // eax
  __int64 *v19; // rax
  __int64 v21; // r9
  const __m128i *v22; // r15
  __m128i *v23; // rax
  const void *v24; // rsi
  char *v25; // r15
  __int64 v26[10]; // [rsp+0h] [rbp-50h] BYREF

  v5 = a4;
  v8 = (__int64)*a1;
  v9 = *((unsigned int *)*a1 + 2);
  v10 = **a1;
  v11 = *(unsigned int *)(v8 + 12);
  v12 = v10 + 24 * v9;
  if ( v9 >= v11 )
  {
    v21 = v9 + 1;
    LODWORD(v26[0]) = a2;
    v22 = (const __m128i *)v26;
    v26[1] = 0;
    v26[2] = v5;
    if ( v11 < v9 + 1 )
    {
      v24 = (const void *)(v8 + 16);
      if ( v10 > (unsigned __int64)v26 || v12 <= (unsigned __int64)v26 )
      {
        sub_C8D5F0(v8, v24, v9 + 1, 0x18u, v11, v21);
        v10 = *(_QWORD *)v8;
        v9 = *(unsigned int *)(v8 + 8);
      }
      else
      {
        v25 = (char *)v26 - v10;
        sub_C8D5F0(v8, v24, v9 + 1, 0x18u, v11, v21);
        v10 = *(_QWORD *)v8;
        v9 = *(unsigned int *)(v8 + 8);
        v22 = (const __m128i *)&v25[*(_QWORD *)v8];
      }
    }
    v23 = (__m128i *)(v10 + 24 * v9);
    *v23 = _mm_loadu_si128(v22);
    v23[1].m128i_i64[0] = v22[1].m128i_i64[0];
    ++*(_DWORD *)(v8 + 8);
  }
  else
  {
    v13 = v9;
    if ( v12 )
    {
      *(_DWORD *)v12 = a2;
      *(_QWORD *)(v12 + 8) = 0;
      *(_QWORD *)(v12 + 16) = v5;
      v13 = *(_DWORD *)(v8 + 8);
    }
    *(_DWORD *)(v8 + 8) = v13 + 1;
  }
  v14 = (__int64 *)a1[2];
  v15 = *(_QWORD *)(*a1[1] + 40);
  v16 = *(_QWORD **)v15;
  v17 = *(unsigned __int8 *)(v15 + 8);
  if ( (unsigned int)(v17 - 17) > 1 )
  {
    sub_BCB2A0(v16);
  }
  else
  {
    v18 = *(_DWORD *)(v15 + 32);
    BYTE4(v26[0]) = (_BYTE)v17 == 18;
    LODWORD(v26[0]) = v18;
    v19 = (__int64 *)sub_BCB2A0(v16);
    sub_BCE1B0(v19, v26[0]);
  }
  return sub_DFD2D0(v14, a2, v15) * a3;
}
