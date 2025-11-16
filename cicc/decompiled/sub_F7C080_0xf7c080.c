// Function: sub_F7C080
// Address: 0xf7c080
//
__int64 __fastcall sub_F7C080(unsigned __int64 **a1, unsigned int a2, unsigned int a3)
{
  __int64 v3; // r8
  __int64 v6; // r12
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rcx
  int v11; // eax
  __int64 v12; // rax
  __int64 *v13; // r14
  _QWORD *v14; // r12
  int v15; // edx
  int v16; // eax
  _QWORD *v17; // rdi
  __int64 *v18; // rax
  __int64 v20; // r8
  const __m128i *v21; // r14
  __m128i *v22; // rax
  __int64 v23; // r9
  char *v24; // r14
  __int64 v25[8]; // [rsp+0h] [rbp-40h] BYREF

  v3 = a3;
  v6 = (__int64)*a1;
  v7 = *((unsigned int *)*a1 + 2);
  v8 = **a1;
  v9 = *((unsigned int *)*a1 + 3);
  v10 = v8 + 24 * v7;
  if ( v7 >= v9 )
  {
    v25[2] = v3;
    v20 = v7 + 1;
    v21 = (const __m128i *)v25;
    LODWORD(v25[0]) = a2;
    v25[1] = 0;
    if ( v9 < v7 + 1 )
    {
      v23 = v6 + 16;
      if ( v8 > (unsigned __int64)v25 || v10 <= (unsigned __int64)v25 )
      {
        sub_C8D5F0(v6, (const void *)(v6 + 16), v7 + 1, 0x18u, v20, v23);
        v8 = *(_QWORD *)v6;
        v7 = *(unsigned int *)(v6 + 8);
      }
      else
      {
        v24 = (char *)v25 - v8;
        sub_C8D5F0(v6, (const void *)(v6 + 16), v7 + 1, 0x18u, v20, v23);
        v8 = *(_QWORD *)v6;
        v7 = *(unsigned int *)(v6 + 8);
        v21 = (const __m128i *)&v24[*(_QWORD *)v6];
      }
    }
    v22 = (__m128i *)(v8 + 24 * v7);
    *v22 = _mm_loadu_si128(v21);
    v22[1].m128i_i64[0] = v21[1].m128i_i64[0];
    ++*(_DWORD *)(v6 + 8);
  }
  else
  {
    v11 = v7;
    if ( v10 )
    {
      *(_DWORD *)v10 = a2;
      *(_QWORD *)(v10 + 8) = 0;
      *(_QWORD *)(v10 + 16) = v3;
      v11 = *(_DWORD *)(v6 + 8);
    }
    *(_DWORD *)(v6 + 8) = v11 + 1;
  }
  v12 = sub_D95540(*(_QWORD *)(*a1[1] + 40));
  v13 = (__int64 *)a1[2];
  v14 = (_QWORD *)v12;
  v15 = *(unsigned __int8 *)(v12 + 8);
  if ( (unsigned int)(v15 - 17) > 1 )
  {
    sub_BCB2A0(*(_QWORD **)v12);
  }
  else
  {
    v16 = *(_DWORD *)(v12 + 32);
    v17 = (_QWORD *)*v14;
    BYTE4(v25[0]) = (_BYTE)v15 == 18;
    LODWORD(v25[0]) = v16;
    v18 = (__int64 *)sub_BCB2A0(v17);
    sub_BCE1B0(v18, v25[0]);
  }
  return sub_DFD2D0(v13, a2, (__int64)v14);
}
