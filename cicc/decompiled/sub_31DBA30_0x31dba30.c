// Function: sub_31DBA30
// Address: 0x31dba30
//
unsigned __int64 __fastcall sub_31DBA30(_QWORD *a1, __int64 a2)
{
  unsigned __int64 result; // rax
  __int64 v4; // r13
  __int64 v5; // rax
  unsigned __int64 v6; // r13
  __int64 v7; // r14
  __int64 v8; // r12
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  int v11; // ecx
  __int64 v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // r8
  __int64 v15; // r9
  int v16; // edi
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // rdx
  const __m128i *v19; // r15
  __m128i *v20; // rax
  __int64 v21; // r15
  __int64 v22; // r13
  unsigned __int64 v23; // r12
  unsigned int v24; // eax
  __int64 v25; // rdi
  __int64 v26; // r8
  char *v27; // r15
  __int64 v28; // [rsp-58h] [rbp-58h] BYREF
  int v29; // [rsp-50h] [rbp-50h]
  __int64 v30; // [rsp-48h] [rbp-48h]
  int v31; // [rsp-40h] [rbp-40h]

  result = *(_QWORD *)(a2 + 8);
  if ( (*(_BYTE *)(result + 878) & 0x40) != 0 )
  {
    v4 = sub_31DA6B0((__int64)a1);
    v5 = sub_31DB4F0((__int64)a1);
    result = sub_E89E30(v4, v5);
    v6 = result;
    if ( result )
    {
      v7 = *(_QWORD *)(a2 + 48);
      if ( !*(_BYTE *)(v7 + 36) )
      {
        v8 = a1[28];
        v9 = *(unsigned int *)(v8 + 128);
        v10 = *(_QWORD *)(v8 + 120);
        v11 = *(_DWORD *)(v8 + 128);
        v12 = 32 * v9;
        if ( (_DWORD)v9 )
        {
          v13 = v10 + v12 - 32;
          v14 = *(_QWORD *)(v13 + 16);
          v11 = *(_DWORD *)(v13 + 24);
          v15 = *(_QWORD *)v13;
          v16 = *(_DWORD *)(v13 + 8);
        }
        else
        {
          v16 = 0;
          v15 = 0;
          v14 = 0;
        }
        v31 = v11;
        v17 = *(unsigned int *)(v8 + 132);
        v18 = v9 + 1;
        v19 = (const __m128i *)&v28;
        v28 = v15;
        v29 = v16;
        v30 = v14;
        if ( v18 > v17 )
        {
          v25 = v8 + 120;
          v26 = v8 + 136;
          if ( v10 > (unsigned __int64)&v28 || (unsigned __int64)&v28 >= v10 + v12 )
          {
            sub_C8D5F0(v25, (const void *)(v8 + 136), v18, 0x20u, v26, v15);
            v10 = *(_QWORD *)(v8 + 120);
            v12 = 32LL * *(unsigned int *)(v8 + 128);
          }
          else
          {
            v27 = (char *)&v28 - v10;
            sub_C8D5F0(v25, (const void *)(v8 + 136), v18, 0x20u, v26, v15);
            v10 = *(_QWORD *)(v8 + 120);
            v19 = (const __m128i *)&v27[v10];
            v12 = 32LL * *(unsigned int *)(v8 + 128);
          }
        }
        v20 = (__m128i *)(v12 + v10);
        *v20 = _mm_loadu_si128(v19);
        v20[1] = _mm_loadu_si128(v19 + 1);
        ++*(_DWORD *)(v8 + 128);
        (*(void (__fastcall **)(_QWORD, unsigned __int64, _QWORD, unsigned __int64, __int64, __int64, __int64, int, __int64, int))(*(_QWORD *)a1[28] + 176LL))(
          a1[28],
          v6,
          0,
          v17,
          v14,
          v15,
          v28,
          v29,
          v30,
          v31);
        v21 = a1[67];
        v22 = a1[28];
        v23 = *(_QWORD *)(v7 + 48) + *(_QWORD *)(v7 + 688);
        v24 = sub_AE4380(a1[25] + 16LL, *(_DWORD *)(a1[25] + 24LL));
        sub_E9A500(v22, v21, v24, 0);
        sub_E98EB0(a1[28], v23, 0);
        return (*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)a1[28] + 168LL))(a1[28]);
      }
    }
  }
  return result;
}
