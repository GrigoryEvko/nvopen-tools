// Function: sub_2E6BF20
// Address: 0x2e6bf20
//
void __fastcall sub_2E6BF20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  bool v6; // zf
  __int64 v7; // rax
  const __m128i *v8; // r13
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // r8
  unsigned __int64 v11; // rdx
  __m128i *v12; // rax
  const void *v13; // rsi
  _BYTE *v14; // r13
  __int64 v15; // [rsp+0h] [rbp-50h] BYREF
  __int64 v16; // [rsp+8h] [rbp-48h]
  __int64 v17; // [rsp+10h] [rbp-40h]
  _BYTE v18[8]; // [rsp+20h] [rbp-30h] BYREF
  __int64 v19; // [rsp+28h] [rbp-28h]
  __int64 v20; // [rsp+30h] [rbp-20h]
  __int64 v21; // [rsp+38h] [rbp-18h]

  if ( *(_QWORD *)(a1 + 544) )
  {
    v6 = *(_BYTE *)(a1 + 560) == 1;
    v15 = a2;
    v16 = a3;
    v17 = a4;
    if ( !v6 )
    {
      sub_2E65A90(a1, &v15, 1, a4, a5, a6);
      if ( !*(_QWORD *)(a1 + 552) )
        return;
LABEL_9:
      sub_2E6B8A0(a1, &v15, 1);
      return;
    }
  }
  else
  {
    if ( !*(_QWORD *)(a1 + 552) )
      return;
    v6 = *(_BYTE *)(a1 + 560) == 1;
    v15 = a2;
    v16 = a3;
    v17 = a4;
    if ( !v6 )
      goto LABEL_9;
  }
  v7 = *(unsigned int *)(a1 + 8);
  v21 = a4;
  v8 = (const __m128i *)v18;
  v9 = *(unsigned int *)(a1 + 12);
  v20 = a3;
  v10 = v7 + 1;
  v18[0] = 1;
  v11 = *(_QWORD *)a1;
  v19 = a2;
  if ( v7 + 1 > v9 )
  {
    v13 = (const void *)(a1 + 16);
    if ( v11 > (unsigned __int64)v18 || (unsigned __int64)v18 >= v11 + 32 * v7 )
    {
      sub_C8D5F0(a1, v13, v10, 0x20u, v10, a6);
      v11 = *(_QWORD *)a1;
      v7 = *(unsigned int *)(a1 + 8);
    }
    else
    {
      v14 = &v18[-v11];
      sub_C8D5F0(a1, v13, v10, 0x20u, v10, a6);
      v11 = *(_QWORD *)a1;
      v7 = *(unsigned int *)(a1 + 8);
      v8 = (const __m128i *)&v14[*(_QWORD *)a1];
    }
  }
  v12 = (__m128i *)(v11 + 32 * v7);
  *v12 = _mm_loadu_si128(v8);
  v12[1] = _mm_loadu_si128(v8 + 1);
  ++*(_DWORD *)(a1 + 8);
}
